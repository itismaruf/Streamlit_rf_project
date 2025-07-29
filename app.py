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


st.subheader("📈 Матрица корреляции между признаками")

corr = df.corr(numeric_only=True)

fig_corr = px.imshow(
    corr,
    text_auto=True,
    color_continuous_scale='RdBu',
    aspect="auto",
    title="Матрица корреляции",
    labels=dict(color="Корреляция")
)
fig_corr.update_layout(width=800, height=600)
st.plotly_chart(fig_corr, use_container_width=True)

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

# 1) Кнопка для первичного обучения
if st.button("🚀 Обучить модель"):
    # разбиение и обучение
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state
    )
    model.fit(X_train, y_train)

    # предсказания и метрики
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)
    test_acc = accuracy_score(y_test, y_pred)
    roc_auc  = roc_auc_score(y_test, y_proba)

    # важность признаков
    feat_imp_df = pd.DataFrame({
        'Признак': X.columns,
        'Важность': model.feature_importances_
    }).sort_values('Важность', ascending=False)

    # график важности
    fig_imp = px.bar(
        feat_imp_df,
        x='Важность', y='Признак',
        orientation='h',
        title="Важность признаков",
        color='Важность',
        color_continuous_scale='Blues'
    )
    fig_imp.update_layout(
        height=500,
        yaxis=dict(autorange="reversed")
    )

    # ROC-кривая
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig_roc = px.area(
        x=fpr, y=tpr,
        title=f"ROC-кривая (AUC={roc_auc:.3f})",
        labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'},
        width=600, height=400
    )
    fig_roc.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    # сохраняем всё в сессии для дальнейшего использования
    st.session_state.trained     = True
    st.session_state.test_acc    = test_acc
    st.session_state.roc_auc     = roc_auc
    st.session_state.feat_imp_df = feat_imp_df
    st.session_state.fig_imp     = fig_imp
    st.session_state.fig_roc     = fig_roc

# 2) После успешного обучения: выводим метрики и графики, и Expander для утечек
if st.session_state.get('trained', False):
    col1, col2 = st.columns(2)
    col1.metric("🎯 Test Accuracy", f"{st.session_state.test_acc:.2%}")
    col2.metric("📈 ROC AUC",       f"{st.session_state.roc_auc:.3f}")

    # визуализация
    st.plotly_chart(st.session_state.fig_imp, use_container_width=True)
    st.plotly_chart(st.session_state.fig_roc, use_container_width=True)

    # expander для выбора признаков-утечек
    with st.expander("💡 Удалить признаки-утечки и переобучить", expanded=True):
        leak_features = st.multiselect(
            "Выберите подозрительные признаки:",
            options=st.session_state.feat_imp_df['Признак'].tolist(),
            key='leaks'
        )
        if leak_features and st.button("🔄 Обучить без утечек"):
            # удаляем утечки и переобучаем
            X_no_leak = X.drop(columns=leak_features)
            X_t2, X_v2, y_t2, y_v2 = train_test_split(
                X_no_leak, y, test_size=0.2, random_state=random_state
            )
            model2 = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=random_state
            )
            model2.fit(X_t2, y_t2)

            y2_proba = model2.predict_proba(X_v2)[:, 1]
            y2_pred  = (y2_proba >= threshold).astype(int)
            acc2     = accuracy_score(y_v2, y2_pred)
            roc2     = roc_auc_score(y_v2, y2_proba)

            # сравнение метрик
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📈 AUC с утечкой",      f"{st.session_state.roc_auc:.3f}")
                st.metric("🎯 Acc с утечкой",      f"{st.session_state.test_acc:.2%}")
            with col2:
                st.metric("🔐 AUC без утечек",    f"{roc2:.3f}")
                st.metric("✅ Acc без утечек",     f"{acc2:.2%}")

            st.success("✅ Модель без утечек обучена и метрики обновлены!")
