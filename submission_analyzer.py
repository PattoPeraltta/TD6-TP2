import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve
import matplotlib.pyplot as plt

# leer el csv con tus predicciones
df = pd.read_csv("modelo_benchmark.csv")

# y_true son los valores reales, y_score las probabilidades predichas
y_true = df["objetivo"] 
y_score = df["pred_proba"]

# calcular distintas métricas
roc_auc = roc_auc_score(y_true, y_score)
acc = accuracy_score(y_true, y_score >= 0.5)   # usando umbral 0.5
prec = precision_score(y_true, y_score >= 0.5)
rec = recall_score(y_true, y_score >= 0.5)
f1 = f1_score(y_true, y_score >= 0.5)

print("roc auc:", roc_auc)
print("accuracy:", acc)
print("precision:", prec)
print("recall:", rec)
print("f1:", f1)

# graficar curva roc
fpr, tpr, _ = roc_curve(y_true, y_score)
plt.plot(fpr, tpr, label=f"roc auc = {roc_auc:.2f}")
plt.plot([0,1],[0,1], linestyle="--", color="gray")
plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
plt.title("curva roc")
plt.legend()
plt.show()
