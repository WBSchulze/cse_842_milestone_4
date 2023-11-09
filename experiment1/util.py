import torch
import numpy as np
import random

def seed_randomness(random_seed):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

def vectors_to_words(vectorizer, vec, threshold):
    # Create a binary vector based on the threshold, indicating significant features.
    binary_vec = (vec > threshold).astype(int)
    
    # Reshape the binary vector to a 2D array for the inverse_transform method.
    binary_vec_2d = binary_vec.reshape(1, -1)
    
    # Convert the binary vector back to words using the vectorizer's vocabulary.
    words = vectorizer.inverse_transform(binary_vec_2d)
    
    return words

def reconstruct(model, samplings, vectorizer, threshold, datapoint_index):
    reconstructed_X = model.decode(samplings)
    reconstructed_vec = reconstructed_X[datapoint_index].cpu().detach().numpy()
    reconstructed_words = vectors_to_words(vectorizer, reconstructed_vec, threshold)
    return reconstructed_words

def perturb_and_reconstruct(model, X_train, X_train_tfidf, X_test_tfidf, vectorizer, threshold, datapoint_index):
    model.eval()
    with torch.no_grad():
        # Get latent samplings for train and test data.
        train_encoding = model.get_latent_encoding(X_train_tfidf)
        test_encoding = model.get_latent_encoding(X_test_tfidf)

        # Save reconstructions before perturbing latent dimensions.
        reconstructed_words_before = reconstruct(model, train_encoding, vectorizer, threshold, datapoint_index)

        # Perturb latent dimensions.
        zeroed_out_dim = 15  # (of 20)
        train_encoding[:, zeroed_out_dim] = 0
        test_encoding[:, zeroed_out_dim] = 0

        # Reconstruct after perturbing latent dimensions.
        reconstructed_words_after = reconstruct(model, train_encoding, vectorizer, threshold, datapoint_index)

        print_results(datapoint_index, X_train, reconstructed_words_before, reconstructed_words_after, zeroed_out_dim)

        return train_encoding, test_encoding

def print_results(datapoint_index, X_train, reconstructed_words_before, reconstructed_words_after, zeroed_out_dim):
    def get_removed_and_added_words(before_words, after_words):
        before_set = set(before_words[0])
        after_set = set(after_words[0])
        removed_words = before_set - after_set
        added_words = after_set - before_set
        return sorted(removed_words), sorted(added_words)

    # Print original, before, and after reconstructions
    print(f"\nOriginal response (i={datapoint_index}):\n{X_train[datapoint_index]}\n")
    print(f"Reconstruction before zeroing out latent dimension (sample {datapoint_index}):\n{reconstructed_words_before}\n")
    print(f"Reconstruction after zeroing out latent dimension (sample {datapoint_index}):\n{reconstructed_words_after}")

    # Compare reconstructions
    removed_words, added_words = get_removed_and_added_words(reconstructed_words_before, reconstructed_words_after)
    print(f"Words removed by zeroing out latent dimension {zeroed_out_dim} (sample {datapoint_index}):\n{removed_words}\n")
    print(f"Words added by zeroing out latent dimension {zeroed_out_dim} (sample {datapoint_index}):\n{added_words}\n")
