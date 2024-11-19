

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
    filtered_words = [word for word 