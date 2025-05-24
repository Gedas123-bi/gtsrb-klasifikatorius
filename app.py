import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models.models import Base, Image, Prediction
from ml.classical_models import SimpleKNN
from ml.cnn_model import SimpleCNN
from utils.charts import generate_charts

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'ppm'}

# Flask konfigūracija
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# SQLAlchemy ryšys
engine = create_engine('sqlite:///gtsrb.db')
Session = sessionmaker(bind=engine)
session = Session()

# Failų tikrinimas
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Pradinis puslapis
@app.route('/')
def index():
    return render_template('upload.html')

# Įkelti nuotrauką
@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['file']
    model_type = request.form.get('model')  # 'cnn' arba 'knn'

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

        # Įrašyti nuotrauką į DB
        new_img = Image(filename=filename, folder=app.config['UPLOAD_FOLDER'], dataset_type='user')
        session.add(new_img)
        session.commit()

        return redirect(url_for('predict_image', filename=filename, model=model_type))
    return "Neteisingas failas. Leidžiami formatai: png, jpg, jpeg, ppm."

# Spėjimas ir rezultato įrašymas
@app.route('/predict/<filename>')
def predict_image(filename):
    try:
        model_type = request.args.get('model', 'knn')  # numatyta: knn
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Paimam treniravimo duomenis
        train_data = session.query(Image).filter_by(dataset_type='train').all()
        if not train_data:
            return "⚠️ Klaida: nėra treniravimo duomenų."

        # Modelio pasirinkimas
        if model_type == 'cnn':
            model = SimpleCNN()
        else:
            model = SimpleKNN()

        # Apmokymas ir prognozė
        model.train(train_data)
        predicted_class = model.predict(img_path)

        # Gauti įrašytą paveikslėlį
        img_obj = session.query(Image).filter_by(filename=filename, folder=app.config['UPLOAD_FOLDER']).first()

        # Įrašyti rezultatą į DB
        prediction = Prediction(
            user='anonimas',
            image_id=img_obj.id,
            model_name=model_type,
            predicted_class=predicted_class,
            true_class=img_obj.class_id if img_obj.class_id is not None else None
        )
        session.add(prediction)
        session.commit()

        return render_template('predict.html', filename=filename, prediction=predicted_class)

    except Exception as e:
        return f"<h3>⚠️ Klaida:</h3><pre>{e}</pre>"
    
@app.route('/charts')
def charts():
    generate_charts()
    return render_template('charts.html')
    
 

# Paleidimas
if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
