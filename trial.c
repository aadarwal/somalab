#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "mongoose.h"  
#include "cJSON.h"     

#define MAX_LENGTH 512
#define MAX_CONVERSATIONS 5
#define MAX_BACKGROUND_LENGTH 1000
#define MAX_STATEMENT_LENGTH 500

typedef struct {
    char patient_background[MAX_BACKGROUND_LENGTH];
    char conversation_history[MAX_CONVERSATIONS][MAX_STATEMENT_LENGTH];
    int conversation_count;
    char doctor_statement[MAX_STATEMENT_LENGTH];
} AnalysisInput;

typedef struct {
    float overall_negativity;
    float perceived_judgment;
    float anxiety_stress;
    float empathy_rapport;
    char rationale[MAX_STATEMENT_LENGTH];
} AnalysisOutput;

typedef struct {
    float weights[4][MAX_LENGTH];
    float biases[4];
} Model;

void prepare_prompt(AnalysisInput *input, char *prompt);
void parse_model_output(float *raw_output, AnalysisOutput *output);
void tokenize(const char *text, float *tokenized, int max_length);
void forward_pass(Model *model, float *input, float *output);
void train_model(Model *model, const char *data_path);
void evaluate_model(Model *model, const char *data_path);
void load_model(Model *model, const char *path);
void save_model(Model *model, const char *path);

static void ev_handler(struct mg_connection *nc, int ev, void *ev_data) {
    struct http_message *hm = (struct http_message *) ev_data;
    
    if (ev != MG_EV_HTTP_REQUEST) return;

    char *json_str = malloc(hm->body.len + 1);
    strncpy(json_str, hm->body.p, hm->body.len);
    json_str[hm->body.len] = '\0';

    cJSON *json = cJSON_Parse(json_str);
    if (!json) {
        mg_printf(nc, "HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n");
        free(json_str);
        return;
    }

    // data extract
    AnalysisInput input;
    cJSON *background = cJSON_GetObjectItemCaseSensitive(json, "patient_background");
    if (cJSON_IsString(background) && (background->valuestring != NULL)) {
        strncpy(input.patient_background, background->valuestring, MAX_BACKGROUND_LENGTH - 1);
    }

    cJSON *statement = cJSON_GetObjectItemCaseSensitive(json, "doctor_statement");
    if (cJSON_IsString(statement) && (statement->valuestring != NULL)) {
        strncpy(input.doctor_statement, statement->valuestring, MAX_STATEMENT_LENGTH - 1);
    }

    char prompt[MAX_LENGTH];
    prepare_prompt(&input, prompt);

    float tokenized_input[MAX_LENGTH] = {0};
    tokenize(prompt, tokenized_input, MAX_LENGTH);

    Model model;
    load_model(&model, "model.bin");

    float raw_output[4];
    forward_pass(&model, tokenized_input, raw_output);

    AnalysisOutput output;
    parse_model_output(raw_output, &output);

    // JSON response
    cJSON *json_response = cJSON_CreateObject();
    cJSON_AddNumberToObject(json_response, "overall_negativity", output.overall_negativity);
    cJSON_AddNumberToObject(json_response, "perceived_judgment", output.perceived_judgment);
    cJSON_AddNumberToObject(json_response, "anxiety_stress", output.anxiety_stress);
    cJSON_AddNumberToObject(json_response, "empathy_rapport", output.empathy_rapport);
    cJSON_AddStringToObject(json_response, "rationale", output.rationale);

    char *json_output = cJSON_Print(json_response);

    //  response
    mg_printf(nc, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n%s",
              (int) strlen(json_output), json_output);

    free(json_str);
    cJSON_Delete(json);
    cJSON_Delete(json_response);
    free(json_output);
}

int main(void) {
    Model model;

    FILE *f = fopen("model.bin", "rb");
    if (f) {
        fclose(f);
        load_model(&model, "model.bin");
    } else {
        train_model(&model, "training_data.csv");
        save_model(&model, "model.bin");
    }

    evaluate_model(&model, "test_data.csv");

    struct mg_mgr mgr;
    struct mg_connection *nc;

    mg_mgr_init(&mgr, NULL);
    nc = mg_bind(&mgr, "8000", ev_handler);
    if (nc == NULL) {
        printf("Failed to create listener\n");
        return 1;
    }

    mg_set_protocol_http_websocket(nc);

    printf("Starting web server on port 8000\n");
    for (;;) {
        mg_mgr_poll(&mgr, 1000);
    }
    mg_mgr_free(&mgr);

    return 0;
}


void prepare_prompt(AnalysisInput *input, char *prompt) {
    snprintf(prompt, MAX_LENGTH, "Patient background: %s\n\nRecent conversation:\n%s\n\n"
             "Analyze the emotional impact of the following doctor statement in this context:\n"
             "\"%s\"\n\nProvide ratings on the following scales:\n"
             "Overall negativity (0-10):\nPerceived judgment/criticism (0-5):\n"