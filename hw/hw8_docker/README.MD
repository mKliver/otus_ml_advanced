### Создание модели 
Что бы сформировать файл модели выполните блокнот `notebooks\model.ipynb`. Файл модели запишется в `service/models/log_reg.joblib`.

### Запуск сервиса в Docker
```
cd service
docker build -t heart_service .
docker run --rm --name heart_service -p 5005:5005 heart_service
```


### Обращение к сервису:
Сервис доступен по адресу `http://127.0.0.1:5005/predict`. Принимает POST запрос с содержимым:

````
{
    "age": 40.0,
    "sex": 1.0,
    "cp": 0.0,
    "trestbps": 152.0,
    "chol": 223.0,
    "fbs": 0.0,
    "restecg": 1.0,
    "thalach": 181.0,
    "exang": 0.0,
    "oldpeak": 0.0,
    "slope": 2.0,
    "ca": 0.0,
    "thal": 3.0
}
````

