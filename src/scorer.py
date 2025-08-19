import json
import pandas as pd
import logging
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier, Pool

logger = logging.getLogger(__name__)

logger.info('Importing pretrained model...')

model = CatBoostClassifier()

model_th = 0.95
logger.info('Pretrained model imported successfully...')

def make_pred(dt, cat_features, path_to_file):
    model.load_model('./models/model_catboost.cbm')
    test_pool = Pool(dt, cat_features=cat_features)
    preds = model.predict_proba(test_pool)[:, 1]
    
    submission = pd.DataFrame({
        'index': pd.read_csv(path_to_file).index,
        'prediction': (preds > model_th).astype(int)
    })

    return submission


def save_feature_importance(output_dir, filename, n=5):
    importances = model.get_feature_importance()
    features = model.feature_names_
    feature_importance = dict(zip(features, importances))
    top_features = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:n])

    with open(f"{output_dir}/{filename}_top_features.json", "w") as f:
        json.dump(top_features, f)


def save_prediction_plot(preds_array, output_dir, filename):
    background_color = '#041744'
    fill_color = "#c9b7f5"
    font_style = 'serif'
    font_color = "#060606"
    plt.figure(figsize=(6,4))
    plt.rcParams['font.family'] = font_style
    plt.title("Плотность распределения предсказаний")
    plt.xlabel("Предсказанная вероятность")
    plt.ylabel("Плотность")
    plt.hist(preds_array, bins=50, density=True, alpha=0.7, color=fill_color)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}_pred_density.png")
    plt.close()