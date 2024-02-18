from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score, roc_auc_score, auc, recall_score
from sklearn.compose import ColumnTransformer
import xgboost as xgb
from imblearn.pipeline import Pipeline
from sklearn.pipeline import Pipeline as Pipeline_skl
from sklearn.preprocessing import OneHotEncoder, RobustScaler
import matplotlib.pyplot as plt
import pandas as pd
import zipfile

def unzip_file(zip_file, extract_to):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        extracted_files = zip_ref.namelist()
    return extracted_files
    

def get_result(search_dict, X_train, y_train, X_test, y_test):
    for model_name, search in search_dict.items():
    
        print('*'*100)
        print(model_name)
        print('*'*100)
        
        best_params = search.best_params_
        best_score = search.best_score_
    
        print("   Best Hyperparameters:", best_params)
        print("   Best Score:", best_score)
    
        print('Training set score: ' + str(search.score(X_train,y_train)))
        print('Test set score: ' + str(search.score(X_test,y_test)))
    
    
        y_pred = search.predict(X_test)
        predicted_probabilities = search.predict_proba(X_test)
        mcc = matthews_corrcoef(y_test, y_pred)
    
        print(classification_report(y_test, y_pred))
        print(f'Matthews Correlation Coefficient: {mcc:.2f}')
        print(f'ROC-AUC Score: {roc_auc_score(y_test, predicted_probabilities[:, 1]):.2f}')

        y_result = pd.concat([pd.DataFrame(y_test), pd.DataFrame(y_pred, index=y_test.index, columns=['isFraud_pred'])], axis=1)
        df_result = pd.concat([X_test['amount'], y_result], axis=1)
        total_fraud_amount_saved = df_result[(df_result.isFraud == 1)  & ((df_result.isFraud_pred == 1))]['amount'].sum()
        print(f'Total saved amount from fraud transactions : £{total_fraud_amount_saved:.2f}')
    
        cm = confusion_matrix(y_test, y_pred, labels=search.classes_)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot()
        plt.title(f'Confusion Matrix of {model_name}')
        plt.show()


def train_xgb(df, numerical_cols = ['step','amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest']):
    X=df.drop('isFraud',axis=1)
    y=df['isFraud']
    X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.7)
    skf=StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    scoring = {
        'f1': make_scorer(f1_score),
        'roc_auc': make_scorer(roc_auc_score),
        'recall': make_scorer(recall_score)
    }

    model = xgb.XGBClassifier(random_state=42)
    
    param = {
            'clf__n_estimators': [100, 200, 250, 300, 350],
            'clf__max_depth': [3, 4, 5],
            'clf__learning_rate': [0.01, 0.1, 0.2, 0.3],
            'clf__subsample': [0.8, 0.9, 1.0],
            'clf__colsample_bytree': [0.8, 0.9, 1.0],
            'clf__gamma': [0, 1, 5],
            'clf__scale_pos_weight': [1, 5, 10, 15],
            'clf__alpha': [0, 0.1, 0.5, 1],
            'clf__lambda': [0, 0.1, 0.5, 1] 
        }

    categorical_cols = ['type']
    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='error'))
    ])
    
    robust_scaler = RobustScaler()
    
    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_cols),
            ("num", robust_scaler, numerical_cols)
        ])
    
    STEPS = [('preprocessor', preprocessor), ("clf", model)]
    pipe = Pipeline_skl(steps=STEPS)
    search = RandomizedSearchCV(pipe, param, cv=skf, n_jobs=-1, scoring=scoring, refit='f1')

    search.fit(X_train, y_train)
    SEARCH_DICT = {}
    SEARCH_DICT[type(model).__name__] = search

    return SEARCH_DICT, X_train, y_train, X_test, y_test, pipe
        