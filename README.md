# ml_Cancer_prediction
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Загрузка данных
df = pd.read_csv('клиндата.csv')

# 🔧 ПОДГОТОВКА ДАННЫХ

# Целевая переменная
y = df['overall_survival']

# 🏥 КЛИНИЧЕСКИЕ ПРИЗНАКИ
clinical_features = [
    'age_at_diagnosis',
    'tumor_size',
    'tumor_stage', 
    'lymph_nodes_examined_positive',
    'mutation_count',
    'nottingham_prognostic_index',
    'chemotherapy',
    'hormone_therapy',
    'radio_therapy',
    'er_status',
    'her2_status',
    'neoplasm_histologic_grade'
]

# 🧬 МУТАЦИОННЫЕ ПРИЗНАКИ
mutation_columns = []
for col in df.columns:
    if col.endswith('_mut') and df[col].dtype in [np.int64, np.float64]:
        mutation_columns.append(col)
    elif col.endswith('_mut'):
        try:
            pd.to_numeric(df[col], errors='coerce')
            mutation_columns.append(col)
        except:
            continue

print(f"🧬 Found {len(mutation_columns)} numeric mutation features")

# 🔄 СОЗДАЕМ ПРИЗНАКИ
X_clinical = df[clinical_features].copy()
X_mutations = df[mutation_columns].copy()
X = pd.concat([X_clinical, X_mutations], axis=1)

print(f"📊 Total features: {X.shape[1]} (Clinical: {len(clinical_features)}, Mutations: {len(mutation_columns)})")

# 🏷️ КОДИРОВАНИЕ КАТЕГОРИАЛЬНЫХ ПЕРЕМЕННЫХ
label_encoders = {}
categorical_cols = ['er_status', 'her2_status']

for col in categorical_cols:
    if col in X.columns:
        X[col] = X[col].astype(str)
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

# 🔧 ПРЕОБРАЗОВАНИЕ В ЧИСЛОВОЙ ФОРМАТ
for col in X.columns:
    if X[col].dtype == 'object':
        try:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        except:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

# 🔧 ОБРАБОТКА ПРОПУЩЕННЫХ ЗНАЧЕНИЙ
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=X.columns)

print(f"✅ Missing values after imputation: {X.isnull().sum().sum()}")

# 🎯 РАЗДЕЛЕНИЕ ДАННЫХ
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\n📈 Data split: Train {X_train.shape[0]}, Test {X_test.shape[0]}")
print(f"🎯 Class distribution - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")

# 🔄 МАСШТАБИРОВАНИЕ ДАННЫХ (очень важно для KNN!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Data scaled for KNN")

# 🎯 ОПТИМИЗАЦИЯ ПАРАМЕТРА K
print("\n🔄 Finding Optimal K Value...")

# Поиск оптимального K
k_range = range(1, 31)
k_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='accuracy')
    k_scores.append(scores.mean())

# Визуализация поиска оптимального K
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(k_range, k_scores, marker='o', linestyle='-', color='blue', linewidth=2, markersize=6)
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Cross-Validation Accuracy')
plt.title('Optimal K Value Selection')
plt.grid(True, alpha=0.3)

# Отмечаем лучший K
best_k = k_range[np.argmax(k_scores)]
best_score = max(k_scores)
plt.axvline(x=best_k, color='red', linestyle='--', alpha=0.7, label=f'Best K = {best_k}')
plt.legend()

print(f"🎯 Optimal K: {best_k} (Accuracy: {best_score:.3f})")

# 🎯 ОБУЧЕНИЕ KNN С ОПТИМАЛЬНЫМ K
knn_model = KNeighborsClassifier(
    n_neighbors=best_k,
    weights='distance',  # Ближайшие соседи имеют больший вес
    metric='minkowski',
    p=2  # Евклидово расстояние
)

print(f"\n🔄 Training KNN with K={best_k}...")
knn_model.fit(X_train_scaled, y_train)

# 📊 ОЦЕНКА МОДЕЛИ
y_pred = knn_model.predict(X_test_scaled)
y_pred_proba = knn_model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
cv_scores = cross_val_score(knn_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\n🎯 KNN MODEL RESULTS")
print("=" * 50)
print(f"✅ Accuracy: {accuracy:.3f}")
print(f"🎯 Cross-val Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
print(f"📊 ROC-AUC: {roc_auc:.3f}")

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Died', 'Survived']))

# 📊 MATRIXA ОШИБОК
plt.subplot(2, 2, 2)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
           xticklabels=['Died', 'Survived'], 
           yticklabels=['Died', 'Survived'])
plt.title('Confusion Matrix - KNN')
plt.ylabel('Actual')
plt.xlabel('Predicted')

# 📊 ROC CURVE
plt.subplot(2, 2, 3)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'KNN (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - KNN')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)

# 🎯 ВИЗУАЛИЗАЦИЯ РАССТОЯНИЙ И СОСЕДЕЙ
print(f"\n🔍 Visualizing KNN neighborhoods...")

# Используем PCA для визуализации в 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)

# Создаем meshgrid для границы решений
h = 0.02  # step size in the mesh
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Обучаем KNN на PCA-признаках для визуализации
knn_viz = KNeighborsClassifier(n_neighbors=best_k)
knn_viz.fit(X_pca, y_train)

# Предсказываем для каждой точки meshgrid
Z = knn_viz.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.subplot(2, 2, 4)
# Рисуем границу решений
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
plt.contour(xx, yy, Z, colors='k', linewidths=0.5, alpha=0.5)

# Рисуем точки данных
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, 
                     cmap=plt.cm.Paired, edgecolors='k', alpha=0.7, s=50)
plt.colorbar(scatter, label='Survival Status')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title(f'KNN Decision Boundary (K={best_k})')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 🔍 АНАЛИЗ БЛИЖАЙШИХ СОСЕДЕЙ ДЛЯ КОНКРЕТНОГО ПАЦИЕНТА
print(f"\n🔍 Analyzing Nearest Neighbors for a Sample Patient...")

# Выбираем случайного пациента из тестовой выборки
sample_idx = np.random.randint(0, len(X_test))
sample_patient = X_test_scaled[sample_idx:sample_idx+1]
sample_actual = y_test.iloc[sample_idx]

# Находим ближайших соседей
distances, indices = knn_model.kneighbors(sample_patient, n_neighbors=best_k)

# Визуализация ближайших соседей
plt.figure(figsize=(15, 5))

# 1. PCA визуализация соседей
plt.subplot(1, 3, 1)
pca_viz = PCA(n_components=2)
X_all_pca = pca_viz.fit_transform(np.vstack([X_train_scaled, X_test_scaled]))

# Разделяем на тренировочные и тестовые точки
train_pca = X_all_pca[:len(X_train_scaled)]
test_pca = X_all_pca[len(X_train_scaled):]

# Визуализируем всех пациентов
plt.scatter(train_pca[:, 0], train_pca[:, 1], c=y_train, 
           cmap='coolwarm', alpha=0.3, s=30, label='All Patients')
plt.scatter(test_pca[sample_idx, 0], test_pca[sample_idx, 1], 
           c='black', marker='*', s=200, label='Target Patient')

# Выделяем ближайших соседей
neighbor_indices = indices[0]
plt.scatter(train_pca[neighbor_indices, 0], train_pca[neighbor_indices, 1],
           c=y_train.iloc[neighbor_indices], cmap='coolwarm', 
           edgecolors='red', linewidths=2, s=100, label='Nearest Neighbors')

plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title(f'KNN Neighbors Visualization\n(K={best_k})')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Расстояния до соседей
plt.subplot(1, 3, 2)
neighbor_distances = distances[0]
neighbor_labels = y_train.iloc[neighbor_indices].values

colors = ['red' if label == 0 else 'green' for label in neighbor_labels]
plt.bar(range(len(neighbor_distances)), neighbor_distances, color=colors, alpha=0.7)
plt.xlabel('Neighbor Index')
plt.ylabel('Distance')
plt.title('Distances to Nearest Neighbors')
plt.xticks(range(len(neighbor_distances)))

# Добавляем легенду
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='green', label='Survived Neighbor'),
                  Patch(facecolor='red', label='Died Neighbor')]
plt.legend(handles=legend_elements)

# 3. Голосование соседей
plt.subplot(1, 3, 3)
vote_counts = pd.Series(neighbor_labels).value_counts()
colors = ['red', 'green']
plt.pie(vote_counts.values, labels=['Died', 'Survived'], colors=colors, 
        autopct='%1.1f%%', startangle=90)
plt.title(f'Neighbor Votes\n(Prediction: {"Survived" if y_pred[sample_idx] == 1 else "Died"})')

plt.tight_layout()
plt.show()

# 📊 АНАЛИЗ ВЛИЯНИЯ РАЗНЫХ K
print(f"\n📊 Analyzing Different K Values...")

k_values = [1, 3, 5, 10, 15, 20, best_k]
k_results = []

plt.figure(figsize=(12, 8))

for i, k in enumerate(k_values):
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train_scaled, y_train)
    y_pred_temp = knn_temp.predict(X_test_scaled)
    accuracy_temp = accuracy_score(y_test, y_pred_temp)
    
    k_results.append({'K': k, 'Accuracy': accuracy_temp})
    
    # Визуализация для разных K
    plt.subplot(2, 4, i+1)
    
    # Упрощенная визуализация границы решений
    if k <= 20:  # Ограничиваем для скорости
        knn_viz_temp = KNeighborsClassifier(n_neighbors=k)
        knn_viz_temp.fit(X_pca, y_train)
        Z_temp = knn_viz_temp.predict(np.c_[xx.ravel(), yy.ravel()])
        Z_temp = Z_temp.reshape(xx.shape)
        
        plt.contourf(xx, yy, Z_temp, alpha=0.3, cmap=plt.cm.Paired)
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, 
                            cmap=plt.cm.Paired, edgecolors='k', alpha=0.5, s=20)
        plt.title(f'K={k}\nAcc: {accuracy_temp:.3f}')
        plt.axis('off')

plt.tight_layout()
plt.suptitle('KNN Decision Boundaries for Different K Values', y=1.02)
plt.show()

k_df = pd.DataFrame(k_results)
print(f"\n📈 Accuracy for Different K Values:")
print(k_df)

# 💡 ФУНКЦИЯ ДЛЯ ИНТЕРАКТИВНОГО ПРОГНОЗА
def knn_patient_analysis(model, scaler, patient_data, patient_id=0, k=5):
    """Детальный анализ пациента с KNN"""
    
    patient_scaled = scaler.transform(patient_data)
    
    # Находим соседей
    distances, indices = model.kneighbors(patient_scaled, n_neighbors=k)
    
    print(f"\n🔍 KNN ANALYSIS FOR PATIENT {patient_id}")
    print("=" * 40)
    print(f"Predicted: {'Survived' if model.predict(patient_scaled)[0] == 1 else 'Died'}")
    print(f"Probability: {model.predict_proba(patient_scaled)[0][1]:.1%}")
    
    print(f"\n📋 {k} NEAREST NEIGHBORS:")
    print("-" * 30)
    
    for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        neighbor_outcome = 'Survived' if y_train.iloc[idx] == 1 else 'Died'
        print(f"{i+1}. Distance: {dist:.3f}, Outcome: {neighbor_outcome}")
    
    return distances, indices

# 📋 ПРИМЕР ИСПОЛЬЗОВАНИЯ
print(f"\n💡 EXAMPLE PATIENT ANALYSIS:")
if len(X_test) > 0:
    sample_patient_data = X_test.iloc[0:1]
    distances, indices = knn_patient_analysis(knn_model, scaler, sample_patient_data, 
                                           patient_id=0, k=best_k)
    print(f"Actual outcome: {'Survived' if y_test.iloc[0] == 1 else 'Died'}")

print(f"\n KNN ANALYSIS COMPLETED!")
print(f" Best K: {best_k}")
print(f" Final Accuracy: {accuracy:.3f}"


<img width="1384" height="928" alt="Screenshot 2026-04-24 162213" src="https://github.com/user-attachments/assets/baa7cc02-3092-4b69-864b-d966bbf45e3b" />


<img width="1409" height="1010" alt="Screenshot 2026-04-24 162237" src="https://github.com/user-attachments/assets/411a4f2a-0693-4f70-a0af-d7941a81f448" />



2-import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Загрузка данных
df = pd.read_csv('клиндата.csv')

# 🔧 ПОДГОТОВКА ДАННЫХ

# Целевая переменная
y = df['overall_survival']

clinical_features = [
    'age_at_diagnosis',
    'tumor_size',
    'tumor_stage', 
    'lymph_nodes_examined_positive',
    'mutation_count',
    'nottingham_prognostic_index',
    'chemotherapy',
    'hormone_therapy',
    'radio_therapy',
    'er_status',
    'her2_status',
    'neoplasm_histologic_grade'
]

mutation_columns = []
for col in df.columns:
    if col.endswith('_mut') and df[col].dtype in [np.int64, np.float64]:
        mutation_columns.append(col)
    elif col.endswith('_mut'):
        try:
            pd.to_numeric(df[col], errors='coerce')
            mutation_columns.append(col)
        except:
            continue

print(f" Found {len(mutation_columns)} numeric mutation features")


X_clinical = df[clinical_features].copy()
X_mutations = df[mutation_columns].copy()
X = pd.concat([X_clinical, X_mutations], axis=1)

print(f" Total features: {X.shape[1]} (Clinical: {len(clinical_features)}, Mutations: {len(mutation_columns)})")

# 🏷️ КОДИРОВАНИЕ КАТЕГОРИАЛЬНЫХ ПЕРЕМЕННЫХ
label_encoders = {}
categorical_cols = ['er_status', 'her2_status']

for col in categorical_cols:
    if col in X.columns:
        X[col] = X[col].astype(str)
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

# 🔧 ПРЕОБРАЗОВАНИЕ В ЧИСЛОВОЙ ФОРМАТ
for col in X.columns:
    if X[col].dtype == 'object':
        try:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        except:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

# 🔧 ОБРАБОТКА ПРОПУЩЕННЫХ ЗНАЧЕНИЙ
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=X.columns)

print(f"Missing values after imputation: {X.isnull().sum().sum()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\n Data split: Train {X_train.shape[0]}, Test {X_test.shape[0]}")
print(f"Class distribution - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Data scaled for SVC")

print("\nTraining Base SVC Model...")

# Базовая модель
base_svc = SVC(
    kernel='rbf',           # Радиальная базисная функция для нелинейных данных
    C=1.0,                  # Параметр регуляризации
    gamma='scale',          # Параметр ядра
    probability=True,       # Включить вероятностные предсказания
    random_state=42
)

base_svc.fit(X_train_scaled, y_train)

# 📊 ОЦЕНКА БАЗОВОЙ МОДЕЛИ
y_pred_base = base_svc.predict(X_test_scaled)
y_pred_proba_base = base_svc.predict_proba(X_test_scaled)[:, 1]

print("\n BASE SVC MODEL RESULTS")
print("=" * 50)

accuracy_base = accuracy_score(y_test, y_pred_base)
cv_scores_base = cross_val_score(base_svc, X_train_scaled, y_train, cv=5, scoring='accuracy')
roc_auc_base = roc_auc_score(y_test, y_pred_proba_base)

print(f"Accuracy: {accuracy_base:.3f}")
print(f"Cross-val Accuracy: {cv_scores_base.mean():.3f} (+/- {cv_scores_base.std() * 2:.3f})")
print(f"ROC-AUC: {roc_auc_base:.3f}")

print("\n Classification Report:")
print(classification_report(y_test, y_pred_base, target_names=['Died', 'Survived']))

# 🔍 ОПТИМИЗАЦИЯ ГИПЕРПАРАМЕТРОВ
print("\nOptimizing SVC Hyperparameters...")

param_grid = {
    'C': [0.1, 1, 10, 100],           # Параметр регуляризации
    'gamma': ['scale', 'auto', 0.01, 0.1, 1],  # Параметр ядра
    'kernel': ['rbf', 'linear']        # Тип ядра
}

grid_search = GridSearchCV(
    SVC(probability=True, random_state=42),
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")

# 🎯 ОБУЧЕНИЕ ОПТИМИЗИРОВАННОЙ МОДЕЛИ
best_svc = grid_search.best_estimator_
y_pred_best = best_svc.predict(X_test_scaled)
y_pred_proba_best = best_svc.predict_proba(X_test_scaled)[:, 1]

accuracy_best = accuracy_score(y_test, y_pred_best)
roc_auc_best = roc_auc_score(y_test, y_pred_proba_best)

print(f"\n OPTIMIZED SVC RESULTS")
print("=" * 50)
print(f" Accuracy: {accuracy_best:.3f}")
print(f" ROC-AUC: {roc_auc_best:.3f}")

print("\n Optimized Classification Report:")
print(classification_report(y_test, y_pred_best, target_names=['Died', 'Survived']))

# 📊 СРАВНЕНИЕ МОДЕЛЕЙ
models_comparison = pd.DataFrame({
    'Model': ['Base SVC', 'Optimized SVC'],
    'Accuracy': [accuracy_base, accuracy_best],
    'ROC-AUC': [roc_auc_base, roc_auc_best]
})

print("\n MODEL COMPARISON:")
print(models_comparison)

# 📊 ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Confusion Matrix - Optimized SVC
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0],
           xticklabels=['Died', 'Survived'], 
           yticklabels=['Died', 'Survived'])
axes[0,0].set_title('Confusion Matrix - Optimized SVC')
axes[0,0].set_ylabel('Actual')
axes[0,0].set_xlabel('Predicted')

# 2. ROC Curves Comparison
fpr_base, tpr_base, _ = roc_curve(y_test, y_pred_proba_base)
fpr_best, tpr_best, _ = roc_curve(y_test, y_pred_proba_best)

axes[0,1].plot(fpr_base, tpr_base, color='blue', lw=2, 
              label=f'Base SVC (AUC = {roc_auc_base:.3f})')
axes[0,1].plot(fpr_best, tpr_best, color='red', lw=2, 
              label=f'Optimized SVC (AUC = {roc_auc_best:.3f})')
axes[0,1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
axes[0,1].set_xlim([0.0, 1.0])
axes[0,1].set_ylim([0.0, 1.05])
axes[0,1].set_xlabel('False Positive Rate')
axes[0,1].set_ylabel('True Positive Rate')
axes[0,1].set_title('ROC Curves Comparison')
axes[0,1].legend(loc="lower right")
axes[0,1].grid(True, alpha=0.3)

# 3. Feature Importance (для линейного ядра)
if grid_search.best_params_['kernel'] == 'linear':
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': np.abs(best_svc.coef_[0])
    }).sort_values('importance', ascending=False)
    
    top_features = feature_importance.head(15)
    sns.barplot(data=top_features, x='importance', y='feature', ax=axes[1,0], palette='viridis')
    axes[1,0].set_title('Feature Importance (Linear SVC)')
    axes[1,0].set_xlabel('Coefficient Magnitude')
else:
    # Для нелинейных ядер используем альтернативный подход
    from sklearn.inspection import permutation_importance
    
    perm_importance = permutation_importance(
        best_svc, X_test_scaled, y_test, n_repeats=10, random_state=42
    )
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': perm_importance.importances_mean
    }).sort_values('importance', ascending=False)
    
    top_features = feature_importance.head(15)
    sns.barplot(data=top_features, x='importance', y='feature', ax=axes[1,0], palette='viridis')
    axes[1,0].set_title('Feature Importance (Permutation)')
    axes[1,0].set_xlabel('Importance Score')

# 4. Model Comparison
models_comparison.plot(x='Model', y=['Accuracy', 'ROC-AUC'], 
                      kind='bar', ax=axes[1,1], color=['skyblue', 'lightcoral'])
axes[1,1].set_title('Model Performance Comparison')
axes[1,1].set_ylabel('Score')
axes[1,1].legend(loc='lower right')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 🔍 АНАЛИЗ ОПОРНЫХ ВЕКТОРОВ
print(f"\n🔍 SUPPORT VECTORS ANALYSIS:")
print(f"Number of support vectors: {best_svc.n_support_}")
print(f"Support vectors per class: {best_svc.n_support_}")

# 💡 ПРОГНОЗИРОВАНИЕ ДЛЯ НОВЫХ ПАЦИЕНТОВ
def predict_survival_svc(model, scaler, patient_data, feature_names=X.columns):
    """Прогноз выживаемости с использованием SVC"""
    # Масштабирование данных пациента
    patient_scaled = scaler.transform(patient_data)
    
    prediction = model.predict(patient_scaled)[0]
    probability = model.predict_proba(patient_scaled)[0][1]
    
    if prediction == 1:
        result = f"🟢 HIGH survival probability: {probability:.1%}"
    else:
        result = f"🔴 LOW survival probability: {1-probability:.1%}"
    
    # Дополнительная информация для линейной модели
    if hasattr(model, 'coef_'):
        feature_effects = pd.DataFrame({
            'feature': feature_names,
            'coefficient': model.coef_[0]
        }).sort_values('coefficient', key=abs, ascending=False)
        
        top_positive = feature_effects.head(3)
        top_negative = feature_effects[feature_effects['coefficient'] < 0].head(3)
        
        result += f"\nTop positive factors: {list(top_positive['feature'])}"
        result += f"\nTop negative factors: {list(top_negative['feature'])}"
    
    return result

# 📋 ПРИМЕР ИСПОЛЬЗОВАНИЯ
print(f"\n💡 EXAMPLE PREDICTION:")
if len(X_test) > 0:
    sample_patient = X_test.iloc[0:1]
    prediction_result = predict_survival_svc(best_svc, scaler, sample_patient)
    print(prediction_result)
    print(f"Actual outcome: {'Survived' if y_test.iloc[0] == 1 else 'Died'}")

# 🧪 ЭКСПЕРИМЕНТ С РАЗНЫМИ ЯДРАМИ
print(f"\n KERNEL COMPARISON:")
kernels = ['linear', 'rbf', 'poly']
kernel_results = []

for kernel in kernels:
    temp_svc = SVC(kernel=kernel, C=1.0, probability=True, random_state=42)
    temp_svc.fit(X_train_scaled, y_train)
    y_pred_temp = temp_svc.predict(X_test_scaled)
    accuracy_temp = accuracy_score(y_test, y_pred_temp)
    kernel_results.append({'Kernel': kernel, 'Accuracy': accuracy_temp})

kernel_df = pd.DataFrame(kernel_results)
print(kernel_df)

print(f"\n🎯 SVC TRAINING COMPLETED!")
print(f" Best model: {grid_search.best_params_}")
print(f" Final Accuracy: {accuracy_best:.3f}")


<img width="1161" height="382" alt="Screenshot 2026-04-24 162029" src="https://github.com/user-attachments/assets/216acb43-cae3-4ba6-8b95-227138e1148a" />
<img width="1330" height="376" alt="Screenshot 2026-04-24 162127" src="https://github.com/user-attachments/assets/7245cb6c-fa38-4282-bef7-71b192ba0afe" />


