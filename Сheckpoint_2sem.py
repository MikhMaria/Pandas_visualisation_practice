#ВАРИАНТ1 (комары)

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


file_name = 'mosquito_Indicator.csv'

# 1. Найти среднемесячные значения показателя количества комаров (mosquito_Indicator). Построить графики для
# исходных и усредненных и месяцам данных.

df = pd.read_csv(file_name)
df['date'] = pd.to_datetime(df['date'])   # обратили в объект типа datetime
df['only_month'] = df['date'].dt.month        #Выделили из даты месяц и закинули в отдельный столбец
# print(df['only_month'])
month_average = df.groupby('only_month')['mosquito_Indicator'].mean()
# print(month_average)

#График для исходных данных
plt.figure(figsize=(14, 6))
plt.plot(df['date'], df['mosquito_Indicator'])
plt.title('Исходные данные по количеству комаров')
plt.xlabel('Дата')
plt.ylabel('Количество комаров')
plt.show()

#График среднего кол-ва по месяцам
plt.figure(figsize=(14, 6))
month_average.plot(kind='bar')  #сделаем столбчатую диаграмму
plt.title('Среднемесячная статистика по кол-ву комаров')
plt.xlabel('Месяц')
plt.ylabel('Среднее количество комаров')
plt.show()

# 2. Найти стандартное отклонение и среднее арифметическое значение количества комаров за весь период.
# Подсчитать число дней в году, когда количество комаров превышает рассчитанное среднее значение, и вывести в
# виде датафрейма с этим количеством по каждому году исходной таблицы.

srednee_arifm_total = df['mosquito_Indicator'].mean().round(2)
print(f"Среднее кол-во за весь период: {srednee_arifm_total}")
standarn_otkl_total = df['mosquito_Indicator'].std().round(2)
print(f"Стандартное отклонение за весь период: {standarn_otkl_total}")

#дни, когда кол-во комаров превышает среднее
df['year'] = df['date'].dt.year
df['above_srednee'] = df['mosquito_Indicator'] > srednee_arifm_total
above_srednee_in_year = df.groupby('year')['above_srednee'].sum()
print('Количество в каждом году дней, когда кол-во комаров выше среднего: ')
print(above_srednee_in_year)


# 3. Построить прогноз численности комаров на следующие 3 года любым доступным способом.

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df['days'] = (df['date'] - df['date'].min()).dt.days #считаем сколько дней прошло от начальной даты

X = df[['days']]
y = df['mosquito_Indicator']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X)

last_date = df['date'].max()
future_dates = [last_date + timedelta(days=i) for i in range(1, 365*3+1)]
future_days = [(date - df['date'].min()).days for date in future_dates]
future_X = pd.DataFrame({'days': future_days})
future_y = model.predict(future_X)



# 4. Визуализировать результаты прогноза: реальные данные и прогноз следует вывести на одном графике и
# выделить разными цветами.


plt.figure(figsize=(14, 6))
plt.plot(df['date'], df['mosquito_Indicator'], label='Реальные данные', color='blue')
plt.plot(future_dates, future_y, label='Прогноз на 3 года', color='red', linestyle='--')
plt.title('Прогноз численности комаров на следующие 3 года')
plt.xlabel('Дата')
plt.ylabel('Количество комаров')
plt.legend()
plt.grid(True)
plt.show()


# 5. Сделать выводы о сезонной динамике численности комаров и интерпретировать полученные прогнозные результаты.

print('Выводы:')
print('Всплески количества комаров приходятся на середину года, то есть примерно на весну-лето')

# 6. Реализовать запрос от пользователя: пользователь вводит дату, программа выводит сообщение со значениями
# в этот день, а также график, где точкой выделена эта дата.

def user_request():
    user_date_str = input("\nВведите дату в формате ГГГГ-ММ-ДД: ")
    try:
        user_date = datetime.strptime(user_date_str, '%Y-%m-%d')
        if user_date in df['date'].values:
            value = df[df['date'] == user_date]['mosquito_Indicator'].values[0]
            print(f"В дату {user_date_str} количество комаров составляло: {value}")
        else:
            if user_date > last_date:
                days = (user_date - df['date'].min()).days
                predicted_value = model.predict([[days]])[0]
                print(f"Прогноз на {user_date_str}: количество комаров составит около {predicted_value:.1f}")
            else:
                print("Для указанной даты нет данных.")


        plt.figure(figsize=(14, 6))
        plt.plot(df['date'], df['mosquito_Indicator'], label='Реальные данные', color='blue')

        if user_date <= last_date:
            plt.scatter([user_date], [df[df['date'] == user_date]['mosquito_Indicator'].values[0]],
                        color='red', s=100, label='Запрошенная дата')
        else:
            plt.scatter([user_date], [predicted_value], color='green', s=100, label='Прогноз на запрошенную дату')

        plt.title(f'Количество комаров на {user_date_str}')
        plt.xlabel('Дата')
        plt.ylabel('Количество комаров')
        plt.legend()
        plt.grid(True)
        plt.show()
    except ValueError:
        print("Ошибка: неверный формат даты. Используйте ГГГГ-ММ-ДД.")

user_request()



