#Jersey number based image retrieval project

##Instructions to use:

1. cd to JerseyNumberDetection
2. Setup a virtual environment and activate it
3. install requirements using the "command pip install -r requirements.txt"
4. if you are using gpu machine then install tensorflow-gpu using the command "pip install tensorflow-gpu==1.14.0"
5. edit the config files train_1.json and train_2.json in configs/ directory

##To retrieve images based on jersey numbers:

python retrieve_by_jersey_number_app.py -q QUERY_NO -s SEARCH_DIR -n NO_IMAGES


##Training:

1.run the command 
  "python train.py -c configs/train_1.json" 
2.after completion run the next command 
  "python train.py -c configs/train_2.json"

##Testing/Evaluating the model:

  "python evaluate.py -c svhn/config.json -w svhn/weights.h5"

                   or
  use JerseyNumberDetection.ipynb notebook to run all 


