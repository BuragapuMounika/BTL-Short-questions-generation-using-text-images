import re
from flask import Flask, render_template, request, session
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data (Uncomment if running for the first time)
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')

# Load Spacy model
nlp = spacy.load("en_core_web_lg")

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'mysecretkey'
app.debug = True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_form', methods=['POST'])
def process_form():
    text = request.form['user_input']
    session['user_input'] = text
    level = int(request.form['question_level'])
    question_count = int(request.form['question_count'])

    questions_and_answers = generate_questions(text, level, question_count)
    questions = [qa[0] for qa in questions_and_answers]
    answers = [qa[1] for qa in questions_and_answers]

    session['gen_quest'] = questions
    session['gen_answ'] = answers

    return render_template('resultPage.html', questions=questions, answers=answers)

def generate_questions(text, level, count1):
    # Tokenize, remove stopwords, and perform POS tagging
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
    pos_tags = nltk.pos_tag(lemmatized_words)

    questions_and_answers = []
    count = 0

    # Generate questions and potential answers based on Bloom's levels
    for word, word_pos in pos_tags:
        if count >= count1:
            break
        question = None
        answer = None

        if word.isalnum() and word_pos.startswith(('NN', 'VB', 'RB', 'JJ')):
            # Level 1: Remember (Recall facts and basic concepts)
            if level == 1:
                if word_pos.startswith('NN'):  # Noun
                    question = f"What is {word.upper()}? Briefly describe its significance."
                    answer = f"{word.upper()} refers to {text}."
                elif word_pos.startswith('VB'):  # Verb
                    question = f"What action does {word.upper()} convey?"
                    answer = f"{word.upper()} indicates the action of {text}."
                elif word_pos.startswith('JJ'):  # Adjective
                    question = f"What quality does {word.upper()} describe?"
                    answer = f"{word.upper()} describes the quality of {text}."
                elif word_pos.startswith('RB'):  # Adverb
                    question = f"In what way does {word.upper()} modify the action?"
                    answer = f"{word.upper()} modifies the action in {text}."

            # Level 2: Understand (Explain ideas or concepts)
            elif level == 2:
                if word_pos.startswith('NN'):
                    question = f"How would you explain {word.upper()} in your own words?"
                    answer = f"{word.upper()} means {text}."
                elif word_pos.startswith('VB'):
                    question = f"Can you explain the function of {word.upper()}?"
                    answer = f"{word.upper()} functions to {text}."
                elif word_pos.startswith('JJ'):
                    question = f"How does {word.upper()} enhance the meaning of {text}?"
                    answer = f"{word.upper()} enhances the meaning by {text}."
                elif word_pos.startswith('RB'):
                    question = f"How does {word.upper()} change the way the action is performed?"
                    answer = f"{word.upper()} modifies the action by {text}."

            # Level 3: Apply (Use information in new situations)
            elif level == 3:
                if word_pos.startswith('NN'):
                    question = f"How would you apply {word.upper()} to a real-world situation?"
                    answer = f"{word.upper()} can be applied in {text}."
                elif word_pos.startswith('VB'):
                    question = f"In what situations would {word.upper()} be used effectively?"
                    answer = f"{word.upper()} is effective in {text}."
                elif word_pos.startswith('JJ'):
                    question = f"How can {word.upper()} be applied to describe other situations?"
                    answer = f"{word.upper()} can be used to describe {text} in different contexts."
                elif word_pos.startswith('RB'):
                    question = f"In what scenarios would {word.upper()} change the outcome of the action?"
                    answer = f"{word.upper()} could change the outcome by {text}."

            # Level 4: Analyze (Draw connections among ideas)
            elif level == 4:
                if word_pos.startswith('NN'):
                    question = f"What are the components of {word.upper()}? How do they relate to {text}?"
                    answer = f"The components of {word.upper()} are {text}."
                elif word_pos.startswith('VB'):
                    question = f"Can you differentiate between the uses of {word.upper()} and similar actions?"
                    answer = f"{word.upper()} differs from other actions by {text}."
                elif word_pos.startswith('JJ'):
                    question = f"How does {word.upper()} influence the meaning of the subject?"
                    answer = f"{word.upper()} changes the way we perceive {text}."
                elif word_pos.startswith('RB'):
                    question = f"What role does {word.upper()} play in changing the meaning of the verb?"
                    answer = f"{word.upper()} alters the verb's meaning by {text}."

            # Level 5: Evaluate (Justify a stand or decision)
            elif level == 5:
                if word_pos.startswith('NN'):
                    question = f"Do you agree with the significance of {word.upper()} in this context? Why or why not?"
                    answer = f"I believe {word.upper()} is significant in this context because {text}."
                elif word_pos.startswith('VB'):
                    question = f"How would you judge the effectiveness of {word.upper()} in achieving {text}?"
                    answer = f"The effectiveness of {word.upper()} is {text}."
                elif word_pos.startswith('JJ'):
                    question = f"Do you agree with the description provided by {word.upper()}? Why or why not?"
                    answer = f"I agree/disagree with the description because {text}."
                elif word_pos.startswith('RB'):
                    question = f"How would you assess the impact of {word.upper()} on the overall meaning?"
                    answer = f"The impact of {word.upper()} is {text}."

            # Level 6: Create (Produce new or original work)
            elif level == 6:
                if word_pos.startswith('NN'):
                    question = f"Can you create a scenario where {word.upper()} plays a crucial role?"
                    answer = f"In a scenario where {word.upper()} plays a crucial role, {text} happens."
                elif word_pos.startswith('VB'):
                    question = f"Can you design an experiment to test the effects of {word.upper()}?"
                    answer = f"An experiment to test {word.upper()} would involve {text}."
                elif word_pos.startswith('JJ'):
                    question = f"Can you invent a situation where {word.upper()} would describe something new?"
                    answer = f"In a new situation, {word.upper()} would describe {text}."
                elif word_pos.startswith('RB'):
                    question = f"Can you propose a new way to modify the action using {word.upper()}?"
                    answer = f"A new way to modify the action with {word.upper()} would be {text}."

            if question and answer:
                questions_and_answers.append((question, answer))
                count += 1

    return questions_and_answers

@app.route('/process_answers', methods=['POST'])
def process_answers():
    quest = session.get('gen_quest', [])
    user_answers = [request.form[f'answer{i}'] for i in range(len(quest))]
    user_text = session.get('user_input', '')

    # Cosine similarity calculation
    vectorizer = TfidfVectorizer()
    vectorized_text = vectorizer.fit_transform([user_text] + user_answers)
    similarity_scores = cosine_similarity(vectorized_text[0:1], vectorized_text[1:]).flatten()

    return render_template('resultPage.html', questions=quest, answers=user_answers, similarity=similarity_scores)

if __name__ == '__main__':
    app.run(debug=True)
