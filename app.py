import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from datetime import datetime

# Загрузка модели и feature names
@st.cache_resource
def load_model():
    pipeline = joblib.load('car_price_pipeline.pkl')
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    return pipeline, feature_names

# Функция для preprocessing пользовательского ввода
def preprocess_input(input_dict, feature_names):
    # Создаем DataFrame с правильными колонками
    input_df = pd.DataFrame([input_dict])
    
    # Добавляем недостающие колонки с NaN
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = np.nan
    
    # Упорядочиваем колонки как в тренировочных данных
    input_df = input_df[feature_names]
    
        
    return input_df

# Основное приложение
def main():
    # Настройка страницы
    st.set_page_config(
    page_title="Предсказание стоимости Авто",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
    )

    
    st.title("🚗 Car Price Prediction App")
    st.markdown("Предсказание стоимости автомобиля на основе его характеристик")
    
    # Загрузка модели
    try:
        pipeline, feature_names = load_model()
        st.success("Модель успешно загружена!")
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        return
    
    with st.form("car_form"):
        st.header("📊 Введите параметры автомобиля")
                
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Основные характеристики")
            vehicle_manufacturer = st.selectbox(
                "Производитель",
                ['HYUNDAI', 'TOYOTA', 'BMW', 'MAZDA', 'NISSAN', 'DODGE', 'MINI',
                'CHEVROLET', 'MITSUBISHI', 'MERCEDES-BENZ', 'LEXUS', 'VOLKSWAGEN',
                'JEEP', 'HONDA', 'FORD', 'OPEL', 'AUDI', 'KIA', 'SUBARU', 'FIAT',
                'LAND ROVER', 'VAZ', 'PORSCHE', 'INFINITI', 'SSANGYONG',
                'ASTON MARTIN', 'JAGUAR', 'SUZUKI', 'LINCOLN', 'CHRYSLER', 'SKODA',
                'VOLVO', 'MG', 'BUICK', 'CADILLAC', 'GREATWALL', 'DAIHATSU',
                'SEAT', 'RENAULT', 'MERCURY', 'FERRARI']
            )
            
            vehicle_year = st.number_input(
                "Год выпуска",
                min_value=1940,
                max_value=datetime.now().year,
                value=2018
            )
            
            current_mileage = st.number_input(
                "Пробег (км)",
                min_value=0,
                max_value=500000,
                value=50000
            )
        
        with col2:
                            
            vehicle_gearbox_type = st.selectbox(
                "Трансмиссия",
                ['Tiptronic', 'Automatic', 'Manual', 'Variator']
            )
        
            doors_cnt = st.selectbox(
                "Количество дверей", 
                [' 2/3', ' 4/5', ' >5']
            )
            
            wheels = st.selectbox(
                "Привод",
                ['Left wheel', 'Right-hand drive']
            )

        with col3:  
            st.subheader("Внешний вид")  
            car_leather_interior = st.selectbox(
                "Кожаный салон",
                [0, 1],
                format_func=lambda x: "Да" if x == 1 else "Нет"
            )
        
            vehicle_color = st.selectbox(
                "Цвет кузова",
                [' Silver ', ' White ', ' Grey ', ' Black ', ' Carnelian red ',
                ' Blue ', ' Sky blue ', ' Golden ', ' Brown ', ' Red ', ' Beige ',
                ' Green ', ' Pink ', ' Orange ', ' Yellow ', ' Purple ']
                )      
                
                
        # Кнопка предсказания
        submitted = st.form_submit_button("🎯 Предсказать стоимость")
    
    with col2:
        st.header("💡 Информация")
        st.info("""
        **Инструкция:**
        1. Заполните все поля формы
        2. Нажмите кнопку 'Предсказать стоимость'
        3. Результат появится ниже
        """)
        
        st.header("📈 Результат")
        result_placeholder = st.empty()
    
    # Обработка формы
    if submitted:
        try:
            # Собираем ввод в словарь
            input_data = {
                'vehicle_manufacturer': vehicle_manufacturer,
                'current_mileage': current_mileage,
                'vehicle_year': vehicle_year,
                'vehicle_gearbox_type': vehicle_gearbox_type,
                'doors_cnt': doors_cnt,
                'wheels': wheels,
                'vehicle_color': vehicle_color,
                'vehicle_interior_color': vehicle_interior_color,
                'car_leather_interior': car_leather_interior,
                
            }
            
            # Preprocessing
            input_df = preprocess_input(input_data, feature_names)
            
            # Предсказание
            prediction = pipeline.predict(input_df)[0]
            
            # Отображаем результат
            with result_placeholder.container():
                st.success(f"### Предсказанная стоимость: ${prediction:,.2f}")
                
                # Дополнительная информация
                st.markdown("---")
                st.subheader("Детали предсказания")
                
                # Визуализация уверенности модели
                st.metric("Предсказанная цена", f"${prediction:,.2f}")
                
                # Показываем введенные данные
                with st.expander("Просмотр введенных данных"):
                    st.json(input_data)
                    
        except Exception as e:
            st.error(f"Ошибка при предсказании: {e}")
    
    # Дополнительная информация в sidebar
    st.sidebar.header("О приложении")
    st.sidebar.info("""
    Это приложение использует модель машинного обучения 
    для предсказания стоимости автомобиля на основе его характеристик.
    
    **Используемые алгоритмы:**
    - Random Forest Regressor
    - Gradient Boosting
    - Предобработка данных
    
    **Данные:** Исторические данные о продажах автомобилей
    """)
    
    st.sidebar.header("Техническая информация")
    st.sidebar.text(f"Количество признаков: {len(feature_names)}")
    st.sidebar.text(f"Модель: {type(pipeline.named_steps['regressor']).__name__}")

if __name__ == "__main__":
    main()