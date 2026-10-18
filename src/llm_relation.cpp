#include "llama.h"

#include <cmath>

#include <iostream>
#include <string>
#include <vector>

namespace
{

void log(enum ggml_log_level level, const char * text, void * user_data)
{
    // NOP
}

float cosine_similarity(std::vector<float> const & a, std::vector<float> const & b)
{
    float dot = 0.0f;
    float norm1 = 0.0f;
    float norm2 = 0.0f;

    for (size_t i = 0; i < a.size(); i++) {
        dot += a[i] * b[i];
        norm1 += a[i] * a[i];
        norm2 += b[i] * b[i];
    }

    if ((norm1 == 0.0f) || (norm2 == 0.0f)) {
        return 0.0f;
    }

    return dot / (std::sqrt(norm1) * std::sqrt(norm2));
}

std::vector<float> get_text_embedding(llama_model * model, std::string const & text)
{
    llama_context_params context_params = llama_context_default_params();
    context_params.embeddings = true;
    context_params.n_ctx = 512;
    context_params.pooling_type = LLAMA_POOLING_TYPE_MEAN;

    llama_context * context = llama_init_from_model(model, context_params);
    if (context == nullptr) {
        std::cerr << "error: failed to create contenxt" << std::endl;
        // ToDo: throw or cleanup
        exit(EXIT_FAILURE);
    }

    llama_vocab const * vocab = llama_model_get_vocab(model);
    int32_t n_embed = llama_model_n_embd(model);

    int32_t n_tokens = -llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, true, true);
    std::vector<llama_token> tokens(n_tokens);
    n_tokens = llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), true, true);

    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); i++) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == tokens.size() - 1);
    }
    batch.n_tokens = tokens.size();

    if (llama_decode(context, batch) != 0) {
        std::cerr << "error: llama_decode: " << text << std::endl;
        // ToDo: cleanup or throw
        exit(EXIT_FAILURE);
    }

    float * raw_embedding = llama_get_embeddings_seq(context, 0);
    if (raw_embedding == nullptr) {
        raw_embedding = llama_get_embeddings(context);
    }
    std::vector<float> embedding(raw_embedding, raw_embedding + n_embed);

    llama_batch_free(batch);
    llama_free(context);

    return embedding;
}

}

int main(int argc, char* argv[])
{
    if (argc <= 4) {
        std::cout << "usage: llm_relation <model> <word> <word> <word>" << std::endl;
        return EXIT_SUCCESS;
    }

    std::string const model_file = argv[1];
    std::string const word_a = argv[2];
    std::string const word_b = argv[3];
    std::string const word_c = argv[4];

    llama_log_set(&log, nullptr);
    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_file.c_str(), model_params);
    if (model == nullptr) {
        std::cerr << "error: failed to load model" << std::endl;
        llama_backend_free();
        return EXIT_FAILURE;
    }

    std::cout << word_a << " - " << word_b << " + " << word_c << std::endl;
    std::vector<float> v_word_a = get_text_embedding(model, word_a);
    std::vector<float> v_word_b = get_text_embedding(model, word_b);
    std::vector<float> v_word_c = get_text_embedding(model, word_c);

    std::vector<float> v_result(v_word_a.size());
    for (size_t i = 0; i < v_word_a.size(); i++) {
        v_result[i] = v_word_a[i] - v_word_b[i] + v_word_c[i];
    }

    std::vector<std::string> candidates = {
        "Queen", "Princess", "Prince", "King", "Man", "Woman", "Emperor", "Castle", "Computer", "Empress"
    };

    std::string best_match = candidates[0];
    std::vector<float> v_candidate = get_text_embedding(model, best_match);
    float max_sim = cosine_similarity(v_result, v_candidate);

    for(auto const & candidate: candidates) {
        v_candidate = get_text_embedding(model, candidate);
        float sim = cosine_similarity(v_result, v_candidate);
        std::cout << candidate << " (" << sim << ")" << std::endl;

        if (sim >= max_sim) {
            max_sim = sim;
            best_match = candidate;
        }
    }

    std::cout << "---" << std::endl;
    std::cout << best_match << " (" << max_sim << ")" << std::endl;

    llama_model_free(model);
    llama_backend_free();
    return EXIT_SUCCESS;
}