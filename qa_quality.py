# import pandas as pd
# import ollama
# import logging
# import time
# from nltk.translate.bleu_score import sentence_bleu
# from rouge_score import rouge_scorer
# import Levenshtein
# from openai import OpenAI
# import httpx


# Configuration
# BASE_URL_LLM = "https://integrate.api.nvidia.com/v1"
# MODEL_LLAMA_3_1_405B = "meta/llama-3.1-405b-instruct"
# NUM_RETRIES = 3
# BASE_SLEEP_TIME = 1


from base import *



# Initialize the Ollama client
# client = ollama.Client()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_combined_prompt(question: str) -> str:
    """
    Create the prompt for the LLM to generate an answer based on the question.
    """
    # v1
    # return (
    #     f"You are an answer generator AI in Azerbaijani based on given questions. Your task is to create answers:\n\n"
    #     f"Question in Azerbaijani:\n{question}\n\n"
    #     f"Provide a clear and accurate answer and include your answer in 1-2 sentences."
    # )

    # v2
    return(
        f"""
            You are an answer generator AI in Azerbaijani based on given questions. Your task is to create answers:

            **Example:**

            **Question in Azerbaijani:** Makroiqtisadiyyat nədir və mikroiqtisadiyyatdan necə fərqlənir?  
            **Generated answer in Azerbaijani:** Makroiqtisadiyyat iqtisadiyyatın böyük miqyasda təhlili ilə məşğul olur, mikroiqtisadiyyat isə kiçik miqyasda, yəni fərdi bazarlarda və şirkətlərdə baş verən prosesləri öyrənir.

            **Your Task:**

            **Question in Azerbaijani:** {question}

            Provide a clear and accurate answer in Azerbaijani and include your answer in 1-2 sentences.
        """
)


def get_answer_from_local_ollama(model: str, question: str) -> str:
    """
    Send a prompt to the local Ollama model and retrieve the answer using the ollama library.
    """
    prompt = create_combined_prompt(question)
    
    stream = ollama.chat(
        model=model,
        messages=[{'role': 'user', 'content': prompt}],
        stream=True
    )
    
    answer = ''
    try:
        for chunk in stream:
            answer += chunk['message']['content']
    except Exception as e:
        logging.error(f"Request to local Ollama failed: {e}")
    
    # if len(answer) > 200:
    #     return ''
    return answer.strip() if answer else "Error"







def get_evaluation_score(question: str, actual_answer: str, predicted_answer: str) -> str:
    """
    Generate an evaluation score between 0 and 100 by comparing the actual and predicted answers.
    """

    # httpx_client = httpx.Client(http2=True, verify=False)

    # Initialize OpenAI client for NVIDIA
    # client = OpenAI(base_url=BASE_URL_LLM, api_key=API_KEY_LLM, http_client=httpx_client)

    #v1
    # prompt = (
    #     f"Evaluate the following answers and provide a score from 0 to 100 based on how well the predicted "
    #     f"answer matches the actual answer based on the asked question. Provide the score only, without any additional text.\n\n"
    #     f"0-10: No answer or completely incorrect\n"
    #     f"11-30: Significant errors or missing key information\n"
    #     f"31-50: Some errors or incomplete information, but recognizable effort\n"
    #     f"51-70: Mostly accurate with minor errors or omissions\n"
    #     f"71-90: Very close to the actual answer with only minor discrepancies\n"
    #     f"91-100: Accurate or nearly perfect match\n\n"
    #     f"Question that asked in Azerbaijani:\n{question}\n\n"
    #     f"Actual Answer in Azerbaijani:\n{actual_answer}\n\n"
    #     f"Predicted Answer in Azerbaijani:\n{predicted_answer}\n\n"
    #     f"Score (0 to 100):"
    # )

    # v2
    prompt = f"""
            Evaluate the following answers and provide a score from 0 to 100 based on how well the predicted
            answer matches the actual answer based on the asked question. Provide the score only, without any additional text.

            0-10: No answer or completely incorrect
            11-30: Significant errors or missing key information
            31-50: Some errors or incomplete information, but recognizable effort
            51-70: Mostly accurate with minor errors or omissions
            71-90: Very close to the actual answer with only minor discrepancies
            91-100: Accurate or nearly perfect match

            **Example:**

            **Question that asked in Azerbaijani:** Makroiqtisadiyyat nədir və mikroiqtisadiyyatdan necə fərqlənir?  
            **Actual Answer in Azerbaijani:** Makroiqtisadiyyat iqtisadiyyatın böyük miqyasda təhlili ilə məşğul olur, mikroiqtisadiyyat isə kiçik miqyasda, yəni fərdi bazarlarda və şirkətlərdə baş verən prosesləri öyrənir.  
            **Predicted Answer in Azerbaijani:** Makroiqtisadiyyat iqtisadiyyatın ümumi aspektlərini öyrənir, mikroiqtisadiyyat isə fərdi bazarları təhlil edir.  
            **Score (0 to 100):** 65

            **Your Task:**

            **Question that asked:** {question}

            **Actual Answer:** {actual_answer}

            **Predicted Answer:** {predicted_answer}

            **Score (0 to 100):**
            """


    # v3
    # prompt = f"""
    #         Evaluate the following answers and provide a score from 0 to 100 based on how well the predicted
    #         answer matches the actual answer based on the asked question. Provide the score only, without any additional text.

    #         0-10: No answer or completely incorrect
    #         11-30: Significant errors or missing key information
    #         31-50: Some errors or incomplete information, but recognizable effort
    #         51-70: Mostly accurate with minor errors or omissions
    #         71-90: Very close to the actual answer with only minor discrepancies
    #         91-100: Accurate or nearly perfect match

    #         **Example:**

    #         0-10: No answer or completely incorrect
    #         **Example:**  
    #         **Question in Azerbaijani** Azərbaycanın paytaxtı hansı şəhərdir?  
    #         **Actual Answer in Azerbaijani:** Bakı  
    #         **Predicted Answer in Azerbaijani:** Yerevan  
    #         **Score (0 to 100):** 5

    #         11-30: Significant errors or missing key information
    #         **Example:**  
    #         **Question in Azerbaijani** Azərbaycanın paytaxtı hansı şəhərdir?  
    #         **Actual Answer in Azerbaijani:** Bakı  
    #         **Predicted Answer in Azerbaijani:** Bakı Azərbaycanda yerləşir, amma paytaxt olduğu qeyd edilmir.  
    #         **Score (0 to 100):** 20

    #         31-50: Some errors or incomplete information, but recognizable effort
    #         **Example:**  
    #         **Question in Azerbaijani** Azərbaycanın paytaxtı hansı şəhərdir?  
    #         **Actual Answer in Azerbaijani:** Bakı  
    #         **Predicted Answer in Azerbaijani:** Bakı Azərbaycanın ən böyük şəhəridir, amma paytaxt olduğunu açıq şəkildə ifadə etmir.  
    #         **Score (0 to 100):** 45

    #         51-70: Mostly accurate with minor errors or omissions
    #         **Example:**  
    #         **Question in Azerbaijani** Azərbaycanın paytaxtı hansı şəhərdir?  
    #         **Actual Answer in Azerbaijani:** Bakı  
    #         **Predicted Answer in Azerbaijani:** Bakı Azərbaycanın paytaxtıdır, həmçinin Qafqaz bölgəsinin ən iri şəhəridir.  
    #         **Score (0 to 100):** 65

    #         71-90: Very close to the actual answer with only minor discrepancies
    #         **Example:**  
    #         **Question in Azerbaijani** Azərbaycanın paytaxtı hansı şəhərdir?  
    #         **Actual Answer in Azerbaijani:** Bakı  
    #         **Predicted Answer in Azerbaijani:** Bakı Azərbaycanın paytaxtı və ən böyük şəhəridir, həmçinin Xəzər dənizi sahilində yerləşir.  
    #         **Score (0 to 100):** 85

    #         91-100: Accurate or nearly perfect match
    #         **Example:**  
    #         **Question in Azerbaijani** Azərbaycanın paytaxtı hansı şəhərdir?  
    #         **Actual Answer in Azerbaijani:** Bakı  
    #         **Predicted Answer in Azerbaijani:** Bakı Azərbaycanın paytaxtı və iqtisadi mərkəzidir, Xəzər dənizi sahilində yerləşir və çoxsaylı tarixi abidələrə malikdir.  
    #         **Score (0 to 100):** 95

    #         **Your Task:**

    #         **Question that asked:** {question}

    #         **Actual Answer:** {actual_answer}

    #         **Predicted Answer:** {predicted_answer}

    #         **Score (0 to 100):**
    #         """


    payload = {
        "model": MODEL_LLAMA_3_1_405B,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 50
    }

    for attempt in range(NUM_RETRIES):
        try:
            completion = client_openai.chat.completions.create(**payload)
            if completion.choices:
                content = completion.choices[0].message.content
                if content:
                    return content.strip()
                logging.error("Content in response is None.")
            else:
                logging.error(f"Unexpected response format: {completion}")
        except Exception as e:
            logging.error(f"Request failed: {e}")
            # if attempt < NUM_RETRIES - 1:
            #     sleep_time = BASE_SLEEP_TIME * (2 ** attempt)
            #     logging.info(f"Retrying in {sleep_time} seconds...")
            #     time.sleep(sleep_time)
            # else:
            #     return "No score received"
    return "Error"



def calculate_bleu_score(actual_answer: str, predicted_answer: str) -> float:
    """
    Calculate BLEU score for the given actual and predicted answers.
    """
    reference = actual_answer.split()
    candidate = predicted_answer.split()
    bleu_score = sentence_bleu([reference], candidate, smoothing_function=SmoothingFunction().method1)
    return 25 * bleu_score

def calculate_rouge_score(actual_answer: str, predicted_answer: str) -> dict:
    """
    Calculate ROUGE score for the given actual and predicted answers.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(actual_answer, predicted_answer)

    normalized_scores = {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }

    return 25 * (0.33 * normalized_scores['rouge1'] + 0.33 * normalized_scores['rouge2'] + 0.33 * normalized_scores['rougeL'])


def calculate_levenshtein_score(actual_answer: str, predicted_answer: str) -> int:
    """
    Calculate Levenshtein distance between actual and predicted answers.
    """
    return 25 * (1 - (Levenshtein.distance(actual_answer, predicted_answer) / len(predicted_answer)))

