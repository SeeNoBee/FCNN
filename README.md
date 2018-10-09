# FCNN
Индивидуальная лабораторная работа по глубокому обучению (Храмов Илья 381706-3м). В данной работе реализована многослойная полносвязная нейронная сеть c *softmax* на последнем слое и функцией потерь *cross-entropy*, заточенная под обучение на наборе данных MNIST.

## Теория
Все необходимые выводы формул находятся в папке *theory*.

## Сборка
Для сборки используется CMake весии не меньше 3.5.0. Настраивать допонительно ничего не трбуется, кроме:
* `FCNN_INSTALL_MNIST_SCRIPT` [включён по умолчанию] - флаг, отвечающий за копирование в установочную директорию скрипта по загрузке набора данных *MNIST* на языке *Python* (*MNIST_download.py*)
* `FCNN_INSTALL_MNIST` [выключен по умолчанию] - флаг, отвечающий за загрузку в установочную директорию набора данных *MNIST* (требуется *Python3* и CMake не менее 3.12, запуск через *MNIST_download.py*)
* `FCNN_MNIST_TRAIN_IMAGES_URL`, `FCNN_MNIST_TRAIN_LABELS_URL`, `FCNN_MNIST_TEST_IMAGES_URL`, `FCNN_MNIST_TEST_LABELS_URL` [по умолчанию указывают на *MNIST* с сайта [yann.lecun.com](http://yann.lecun.com/exdb/mnist "MNIST")] - ссылки на скачивание файлов тренировочных изображений, тренировочных меток, тестовых изображений и тестовых меток соответственно (появляются только при включённом флаге `FCNN_INSTALL_MNIST`, передаются потом в *MNIST_download.py* в автоматическом режиме)

Для полного функционировния необходимо также проинсталлировать проект. Путь утановки (`CMAKE_INSTALL_PREFIX`) по умолчанию настроен на `build/bin` в корневой директории проекта (папка с *CMakeLists.txt*).
## Запуск
Для запуска рядом с исполняемым файлом должны лежать файлы набора данных *MNIST* со следующимим именами:
1. *train-images.idx3-ubyte* - тренировочные картинки
2. *train-labels.idx1-ubyte* - тренировочные метки
3. *t10k-images.idx3-ubyte* - тестовые картинки
4. *t10k-labels.idx1-ubyte* - тестовые метки

Положить их можно вручную, но если был активен флаг `FCNN_INSTALL_MNIST_SCRIPT`, то рядом с исполняемым файлом будет лежать скрипт *MNIST_download.py*. При его запуске в качестве аргументов можно передать ссылки на файлы *MNIST* в порядке, как в списке выше, а если запустить без аргументов, то *MNIST* будет скачан с сайта [yann.lecun.com](http://yann.lecun.com/exdb/mnist "MNIST"). Скрипт скачает архивы, распакует их и присвоит файлам нужные имена автоматически. При активном флаге `FCNN_INSTALL_MNIST` скрипт будет запущен автоматически на этапе инсталляции проекта, а весь вывод скрипта будет записан в файл *MNISTDownload.cmake*.
### Формат запуска
```
FCNN [скорость_обучения] [количество_эпох] [количество_нейронов_в_скрытом_слое_1 количество_нейронов_в_скрытом_слое_2 ...]
```
К примеру запуск обучения двуслойной сети с 20 нейронами на скрытом слое со скоростью 0.01 на 10 эпохах производится следующим образом:
```
FCNN 0.01 10 20
```
В консоли будет показан прогрэсс обучения на конкретной эпохе, а вконце будет выведена точность в процентах.

