#include <llama.h>

#include <iostream>
#include <string>
#include <vector>

namespace
{

void log(enum ggml_log_level level, const char * text, void * user_data)
{
    // NOP
}

}

int main(int argc, char* argv[])
{
    if (argc <= 2) {
        std::cout << "usage: single_prompt <model> <prompt>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string const model_file = argv[1];
    std::string const prompt = argv[2];

    llama_log_set(&log, nullptr);
    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_file.c_str(), model_params);
    if (model == nullptr) {
        std::cerr << "error: failed to load model" << std::endl;
        llama_backend_free();
        return EXIT_FAILURE;
    }

    llama_chat_message message = {"user", prompt.c_str()};

    char formatted_chat[1024];

    char const * chat_template = llama_model_chat_template(model, nullptr);
    if (chat_template == nullptr) {
        std::cerr << "error: missing chat template in model" << std::endl;
        return EXIT_FAILURE;
    }

    int32_t res = llama_chat_apply_template(
        chat_template,
        &message,
        1,
        true,
        formatted_chat,
        1024
    );
    if (res < 0) {
        std::cerr << "error: failed to apply chat template" << std::endl;
        return EXIT_FAILURE;
    }

    if (res > 1024) {
        std::cerr << "error: failed to apply chat template" << std::endl;
        return EXIT_FAILURE;
    }

    llama_context_params context_params = llama_context_default_params();
    //context_params.embeddings = true;
    context_params.n_ctx = 512;
    context_params.pooling_type = LLAMA_POOLING_TYPE_MEAN;

    llama_context * context = llama_init_from_model(model, context_params);
    if (context == nullptr) {
        std::cerr << "error: failed to create contenxt" << std::endl;
        return EXIT_FAILURE;
    }

    llama_vocab const * vocab = llama_model_get_vocab(model);
    int32_t n_embed = llama_model_n_embd(model);

    int32_t n_tokens = -llama_tokenize(vocab, formatted_chat, res, nullptr, 0, true, true);
    std::vector<llama_token> tokens(n_tokens);
    n_tokens = llama_tokenize(vocab, formatted_chat, res, tokens.data(), tokens.size(), true, true);

    llama_batch batch = llama_batch_init(n_tokens, 0, 1);

    for (int32_t i = 0; i < n_tokens; i++) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == n_tokens - 1);
    }
    batch.n_tokens = n_tokens;

    if (llama_decode(context, batch) != 0) {
        std::cerr << "error: failed to decode batch" << std::endl;
        return EXIT_FAILURE;
    }

    llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
    llama_sampler * sampler = llama_sampler_chain_init(sampler_params);
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.7f)); // temperature
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(1234)); // random seed

    int n_cur = batch.n_tokens;
    int n_max_predict = 512;

    while (n_cur < (n_max_predict + batch.n_tokens)) {
        llama_token new_token_id = llama_sampler_sample(sampler, context, -1);

        if (llama_vocab_is_eog(vocab, new_token_id)) {
            break;
        }

        char buf[256];
        int32_t n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
        if (n > 0) {
            std::string piece(buf, n);
            std::cout << piece << std::flush;
        }

        batch = llama_batch_get_one(&new_token_id, 1);
        
        if (llama_decode(context, batch) != 0) {
            std::cerr << "error: failed to decode" << std::endl;
            return EXIT_FAILURE;
        }

        n_cur++;
    }
    std::cout << std::endl;

    llama_sampler_free(sampler);
    llama_free(context);
    llama_model_free(model);
    llama_backend_free();

    return EXIT_SUCCESS;
}