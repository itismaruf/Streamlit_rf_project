import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

st.set_page_config(page_title="Titanic Data Overview", layout="wide")
st.title("🚢 Titanic - Анализ данных")

st.subheader("Загрузка данных")
df = pd.read_csv("cleaned_titanic.csv")

st.write("Случайные 5 строк из датасета:")
st.dataframe(df.sample(5), use_container_width=True)

st.write(f"🧾 Размер данных: {df.shape[0]} строк и {df.shape[1]} колонок")

st.subheader("📈 Статистическое описание данных")
st.dataframe(df.describe(), use_container_width=True)

st.subheader("🔍 Визуализация признаков")

fig_sex = px.histogram(df, x="Sex", color="Survived", barmode="group",
                       title="Выживание по полу")
st.plotly_chart(fig_sex, use_container_width=True)

fig_age = px.histogram(df, x="Age", color="Survived", nbins=30,
                       title="Распределение возраста и выживаемость")
st.plotly_chart(fig_age, use_container_width=True)


st.subheader("🤖 Обучение модели")

with st.expander("⚙️ Настройки модели", expanded=True):
    n_estimators = st.slider("Количество деревьев (n_estimators)", 10, 500, 100, step=10)
    max_depth = st.slider("Максимальная глубина дерева (max_depth)", 1, 20, 5)
    min_samples_split = st.slider("Минимум образцов для сплита (min_samples_split)", 2, 10, 2)
    random_state = st.number_input("Random state", value=42, step=1)
    threshold = st.slider("Порог классификации (threshold)", 0.0, 1.0, 0.5, step=0.01)

features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked']
df = df[features + ['Survived']]
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

X = df.drop("Survived", axis=1)
y = df["Survived"]

if st.button("🚀 Обучить модель"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state
    )
    model.fit(X_train, y_train)

    # Предсказания вероятностей
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # Применение порога
    y_train_pred = (y_train_proba >= threshold).astype(int)
    y_test_pred = (y_test_proba >= threshold).astype(int)

    # Метрики
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    roc_auc = roc_auc_score(y_test, y_test_proba)

    col1, col2, col3 = st.columns(3)
    col1.metric("🎯 Train Accuracy", f"{train_acc:.2%}")
    col2.metric("🧪 Test Accuracy", f"{test_acc:.2%}")
    col3.metric("📈 ROC AUC", f"{roc_auc:.3f}")

    st.success("✅ Модель успешно обучена!")

    # ROC-кривая
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    fig = px.area(
        x=fpr, y=tpr,
        title=f"ROC-кривая (AUC={roc_auc:.3f})",
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=600, height=400
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    st.plotly_chart(fig)

else:
    st.info("Установите параметры и нажмите кнопку для обучения модели.")
