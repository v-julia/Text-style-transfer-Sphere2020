# Text-style-transfer-Sphere2020

Проект посвящен переносу стилей текстов.

StyleTransfer.pptx - презентация проекта.

1) Перенос стилей стихов Пушкина и Цветаевой

- `1_LoadingPoems.ipynb` - загрузка стихотворений с сайта http://stih.su/
- `2_SplittingToSentences.ipynb` - добработка стихотворений и разбиение на предложения
- `3_Poems_Models.ipynb` - обучение классификатора FastText на стихах

2) Твиты с положительной и отрицательной тональностью (http://study.mokoron.com/)

- `DataTwitter_final.ipynb` - обработка твитов, обучение классификатора FastText

Для того, чтобы запустить обучение нейросети, надо скачать классификаторы стилей и языковые модели (https://drive.google.com/file/d/10AQgKntEatBIQcJNm0ATYGMjCOeABZmP/view?usp=sharing), распаковать их в папку evaluator. Для обучения нейросети на датасете со стихотворениями, запустить `main.py`, с русскими твитами - `main_twitter.py`.
