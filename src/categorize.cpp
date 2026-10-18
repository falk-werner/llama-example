#include <llama.h>
#include <nlohmann/json.hpp>

#include <iostream>
#include <fstream>

using json = nlohmann::json;

namespace
{

void log(enum ggml_log_level level, const char * text, void * user_data)
{
    // NOP
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

}

int main(int argc, char * argv[])
{
    if (argc <= 3) {
        std::cout << "categorize <model> <categories> <subject>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string const model_file = argv[1];
    std::string const categories_file = argv[2];
    std::string const subject = argv[3];

    llama_log_set(&log, nullptr);
    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_file.c_str(), model_params);
    if (model == nullptr) {
        std::cerr << "error: failed to load model" << std::endl;
        llama_backend_free();
        return EXIT_FAILURE;
    }

    auto v_subject = get_text_embedding(model, subject);

    std::ifstream infile(categories_file);
    json data = json::parse(infile);

    for(auto const &category: data["categories"]) {
        auto const v_category = get_text_embedding(model, category["description"]);
        float sim = cosine_similarity(v_subject, v_category);

        std::string const name = category["name"];
        std::cout << name << " (" << sim << ")" << std::endl;
    }

    llama_model_free(model);
    llama_backend_free();
    return EXIT_SUCCESS;
}

