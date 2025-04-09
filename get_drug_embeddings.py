from process_data import *

if __name__ == "__main__":
    from data_utils.get_GEM_embedding import get_gem_embeding
    print("get drug embeddings")
    # get drug embeddings
    get_gem_embeding(initial_datasets, drug_embedding_path, drug_embedding_log, initial_datasets)