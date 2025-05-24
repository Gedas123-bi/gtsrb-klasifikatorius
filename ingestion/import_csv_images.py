import os
import csv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models.models import Base, Class, Image

engine = create_engine('sqlite:///gtsrb.db')
Session = sessionmaker(bind=engine)
session = Session()

# Įkeliame klases (0–42)
for i in range(43):
    session.merge(Class(id=i, name=str(i)))
session.commit()

# treniruojame
train_base = 'data/GTSRB/Final_Training/Images'
for class_folder in os.listdir(train_base):
    class_path = os.path.join(train_base, class_folder)
    if not os.path.isdir(class_path):
        continue
    for img_file in os.listdir(class_path):
        if img_file.endswith(".ppm"):
            session.add(Image(
                filename=img_file,
                folder=class_path,
                class_id=int(class_folder),
                dataset_type='train'
            ))
session.commit()

# testuojame
test_folder = 'data/GTSRB/Final_Test/Images'
gt_path = 'data/GTSRB/Final_Test/GT-final_test.csv'  # Tikslus failo pavadinimas

with open(gt_path, 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    next(reader)  # Praleidžiame pirmą eilutę (antraštę)
    for row in reader:
        filename = row[0]
        class_id = int(row[7])  # 8-tas stulpelis: klasė
        session.add(Image(
            filename=filename,
            folder=test_folder,
            class_id=class_id,
            dataset_type='test'
        ))
session.commit()
print("✅ Duomenys importuoti į DB.")
