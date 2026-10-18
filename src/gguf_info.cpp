#include <llama.h>
#include <nlohmann/json.hpp>

#include <iostream>

using json = nlohmann::json;

namespace
{

void log(enum ggml_log_level level, const char * text, void * user_data)
{
    // NOP
}

}

int main(int argc, char* argv[])
{
    if (argc <= 1) {
        std::cout << "usage: " << argv[0] << " <filename>" << std::endl;
        return EXIT_SUCCESS;
    }

    std::string const model_file = argv[1];

    llama_log_set(&log, nullptr);
    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_file.c_str(), model_params);
    if (model == nullptr) {
        std::cerr << "error: failed to load model" << std::endl;
        llama_backend_free();
        return EXIT_FAILURE;
    }

    json out;
    
    json meta;
    int32_t meta_count = llama_model_meta_count(model);
    for (int32_t i = 0; i < meta_count; i++) {
        char key[256];
        int ret = llama_model_meta_key_by_index(model, i, key, 256);
        if (ret > 0) {
            char val[256];
            ret = llama_model_meta_val_str(model, key, val, 256);
            if (ret > 0) {
                meta[key] = val;
            }
        }
    }
    out["meta"] = meta;

    out["parameter_count"] = llama_model_n_params(model);
    out["model_size"] =  llama_model_size(model);
    out["context_size"] = llama_model_n_ctx_train(model);

    llama_vocab const * vocab = llama_model_get_vocab(model);
    out["vocab_size"] = llama_vocab_n_tokens(vocab);

    char const * chat_template = llama_model_chat_template(model, nullptr);
    if (chat_template != nullptr) {
        out["chat_template"] = chat_template;
    }

    std::cout << out.dump(4) << std::endl;

    llama_model_free(model);
    llama_backend_free();
    return EXIT_SUCCESS;
}
