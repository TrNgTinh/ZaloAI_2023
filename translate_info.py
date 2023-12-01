from Translate_Vi2En.vi2en import TranslatorModule
import pandas as pd
import time
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np
import faiss

def build_faiss_index_gpu(df, column):
    # Assuming df[column] contains embeddings as a list of arrays
    embeddings = np.array(df[column].tolist(), dtype=np.float32)
    
    # If embeddings have shape (N, 1, M), squeeze the second dimension
    embeddings = np.squeeze(embeddings, axis=1)
    
    # Normalize embeddings
    faiss.normalize_L2(embeddings)
    
    # Use GPU resources
    res = faiss.StandardGpuResources()
    
    # Build the FAISS GPU index
    index = faiss.GpuIndexFlatIP(res, len(embeddings[0]))
    index.add(embeddings)
    
    return index

def build_faiss_index(df, column):
    # Assuming df[column] contains embeddings as a list of arrays
    embeddings = np.array(df[column].tolist(), dtype=np.float32)
    
    # If embeddings have shape (N, 1, M), squeeze the second dimension
    embeddings = np.squeeze(embeddings, axis=1)
    
    # Normalize embeddings
    faiss.normalize_L2(embeddings)
    
    # Build the FAISS index
    index = faiss.IndexFlatIP(len(embeddings[0]))
    index.add(embeddings)
    
    return index
    
def find_top_similar_sentences(user_sentence, df, model, tokenizer, top_n=5):
    user_vector = embed_sentence(user_sentence, model, tokenizer)
    series_vectors = df['Embedding'].tolist()

    # Calculate cosine similarity between user input and each sentence in the series
    similarities = [cosine_similarity(user_vector, vector.reshape(1, -1))[0][0] for vector in series_vectors]

    # Get the indices of the top N most similar sentences
    top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:top_n]

    return top_indices

def embed_sentence(sentence, model, tokenizer, device = "cuda:0"):
    input_ids = tokenizer.encode(sentence, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(input_ids)
    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return embeddings

def embed_single_value(item, model, tokenizer, default_embedding=np.zeros(shape=(1, 768))):
    return embed_sentence(item, model, tokenizer) if pd.notna(item) else default_embedding
    
def create_embedding_column(df, name_col_in, name_col_out,  model, tokenizer, default_embedding=np.zeros(shape = (1,768))):
    # Create a new column 'Embedding' in the DataFrame
    df[name_col_out] = df[name_col_in].apply(lambda s: embed_sentence(s, model, tokenizer) if pd.notna(s) else default_embedding)
    return df

def relatedness_fn(index_build, query_embedding,  k=1):
        faiss.normalize_L2(query_embedding)
        query_embedding = np.array(query_embedding, dtype=np.float32)
        # Use Faiss to find the k-nearest neighbors
        D, I = index_build.search(query_embedding.reshape(1, -1), k)
        return D[0], I[0]

def search_faiss_index_gpu(faiss_index, query_embedding, k=1):
    # Assuming query_embedding is the embedding of the query
    query_embedding = np.array(query_embedding, dtype=np.float32)
    
    # Normalize the query embedding
    faiss.normalize_L2(query_embedding)
    
    # Use GPU resources
    res = faiss.StandardGpuResources()
    
    # Convert the query embedding to GPU
    query_embedding_gpu = faiss.float_vector_to_gpu(query_embedding.reshape(1, -1), res)
    query_embedding_gpu = faiss.StandardGpuResources().from_numpy(query_embedding.reshape(1, -1))
    
    # Perform the search on GPU
    D, I = faiss.knn_gpu(res, query_embedding_gpu, faiss_index, k)
    
    # Convert the results back to CPU
    D = D.reshape(-1)
    I = I.reshape(-1)
    
    return D, I


class TranslatorProcessor:
    def __init__(self, input_csv_file, categorical_file, device="cuda:0", cache_dir = "cache"):
        self.translator = TranslatorModule(device=device)
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base-v2", cache_dir = cache_dir).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2", cache_dir = cache_dir)
        self.df = pd.read_csv(input_csv_file)
        self.df_cate = pd.read_csv(categorical_file, delimiter='\t', quoting=3)  # quoting=3 means QUOTE_NONE, which disables quoting
        self.df_cate = self.df_cate[~self.df_cate.duplicated(subset='loc')]
        self.df_cate = create_embedding_column(self.df_cate, "loc", "loc_emb", self.phobert, self.tokenizer)
        self.index_build = build_faiss_index(self.df_cate, "loc_emb")
        
    def categorical_row(self, row, categorical_file, out_csv_file, col_emb = 'caption', col_translate = 'caption'):
        #Categorical
        #Raw Df
        item = row[col_emb]
        emb = embed_single_value(item, self.phobert, self.tokenizer)

        cosinsimilarity, index = relatedness_fn(self.index_build, emb)  

        classes = self.translator.translate_vi2en(self.df_cate.iloc[index[0]]['loc'])
        
        row['score'] = cosinsimilarity[0]
        row['Categorical'] = classes[0]

        row = self.translate_row(row, col_translate)
        
        return row
           
    def categorical(self, csv_file, categorical_file, out_csv_file):
        #Categorical
        df_cate = pd.read_csv(categorical_file, delimiter='\t', quoting=3)  # quoting=3 means QUOTE_NONE, which disables quoting
        df_cate = df_cate[~df_cate.duplicated(subset='loc')]
        df_cate = create_embedding_column(df_cate, "loc", "loc_emb", self.phobert, self.tokenizer)

        index_build = build_faiss_index(df_cate, "loc_emb")

        #Raw Df
        df = pd.read_csv(csv_file)
        df = create_embedding_column(df, "caption", "caption_emb", self.phobert, self.tokenizer)

        cosinsimilarities = []
        indexes = []
        # Duyệt qua từng hàng trong DataFrame
        for i, row in df.iterrows():
            embedding = row['caption_emb']
            #cosinsimilarity, index = search_faiss_index_gpu(index_build, embedding)  
            cosinsimilarity, index = relatedness_fn(index_build, embedding)  
            cosinsimilarities.append(cosinsimilarity[0])

            classes = self.translator.translate_vi2en(df_cate.iloc[index[0]]['loc'])
            indexes.append(classes[0])
            
        # Tạo cột mới 'score' và 'Categorical'
        df['score'] = cosinsimilarities
        df['Categorical'] = indexes

        # Hiển thị DataFrame kết quả
        df = df.drop('caption_emb', axis=1)
        df.to_csv(out_csv_file, index=False)

    def translate_row(self, row, col_translate = 'caption'):  
        row[col_translate + "_en"] = self.translate_if_needed(row[col_translate])
        return row
    
    def translate_and_save_to_csv(self, input_csv_file, output_csv_file, col_translate):
        # Load data from CSV
        data = pd.read_csv(input_csv_file)

        # Apply translation to the specified column
        data[col_translate + "_en"] = data[col_translate].apply(self.translate_if_needed)

        # Save the translated data to a new CSV file
        data.to_csv(output_csv_file, index=False)

    def translate_if_needed(self, text):
        if not pd.isnull(text) and text.strip() != "":
            translated_text = self.translator.translate_vi2en(text)
            return translated_text[0] if translated_text else ""
        return text


# Example usage
if __name__ == "__main__":
    
    start_time = time.time() 
    
    input_csv_file = '/data/tinhtn/Banner/Zalo/Data/test/info.csv'
    output_csv_file = '/data/tinhtn//Banner/Zalo/Data/test/info_thuoc_en_phan_loai.csv'
    cate_file = '/data/tinhtn/Banner/Zalo/Data/test/loc.txt'

    col_translate = "caption"

    t1 = time.time() 
    translator_processor = TranslatorProcessor(input_csv_file, cate_file)
    t2 = time.time() 
    print("Init time", t2 - t1)

    t3 = time.time() 

    df = pd.read_csv(input_csv_file)
    row_1 = df.iloc[1]
    print(translator_processor.categorical_row(row_1, cate_file, output_csv_file))
    t4 = time.time() 
    print("Time phan loai", t4 - t3)

    
    #translator_processor.translate_and_save_to_csv(output_csv_file, output_csv_file, col_translate)
    t5 = time.time() 

    print("Time Dich", t5 - t4)
    
    end_time = time.time()

    print("Time:", end_time - start_time)
