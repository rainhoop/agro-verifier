# Сторонние компоненты и данные

## Геоданные и внешние сервисы

- **Copernicus Sentinel-2 L2A** — спутниковые данные Copernicus. Обязательная
  атрибуция для опубликованных обработанных данных: **Contains modified
  Copernicus Sentinel data 2026**.
- **Copernicus Data Space Ecosystem (CDSE)** — каталог данных и статистический
  API. Использование регулируется условиями CDSE и лицензией исходных данных.
- **Open-Meteo** — погодный контекст. Данные распространяются по
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); указывайте Open-Meteo
  и сохраняйте ссылку на источник при повторном использовании.
- **OpenStreetMap** — данные карты доступны по
  [ODbL](https://opendatacommons.org/licenses/odbl/); требуется атрибуция
  OpenStreetMap contributors.
- **Nominatim** — геокодирование OpenStreetMap. Соблюдайте
  [политику использования сервиса](https://operations.osmfoundation.org/policies/nominatim/),
  включая требования к частоте и атрибуции.
- **Esri World Imagery** — спутниковая подложка предоставляется на условиях
  Esri. Сохраняйте видимую атрибуцию: **Tiles © Esri — Source: Esri, Maxar,
  Earthstar Geographics**. Проверяйте условия использования Esri перед
  повторным распространением снимков.
- **Sentinel-2 cloudless by EOX** — в текущем коде и конфигурации не используется.

## Программные компоненты

Версии Python-пакетов указаны по `requirements.txt` и `requirements-dev.txt`.
Лицензии приводятся по метаданным и лицензиям проектов; при сомнениях
проверяйте репозиторий соответствующего проекта.

| Компонент | Версия | Лицензия |
|---|---:|---|
| FastAPI | 0.162.1 | MIT |
| Uvicorn | 0.53.0 | BSD-3-Clause |
| NumPy | 2.5.3 | BSD-3-Clause |
| scikit-learn | 1.9.1 | BSD-3-Clause |
| joblib | 1.6.0 | BSD-3-Clause |
| Pydantic | 2.13.5 | MIT |
| HTTPX | 0.28.1 | BSD-3-Clause |
| python-multipart | 0.0.32 | Apache-2.0 |
| pytest | 9.2.0 | MIT |
| fontTools | 4.67.0 | MIT |
| Brotli | 1.2.1 | MIT |
| Leaflet | 1.9.4 | BSD-2-Clause |
| Leaflet.draw | vendor-файл 1.0.4 | MIT; см. репозиторий проекта |
| Chart.js | 4.4.7 | MIT |
| Inter | vendor-файлы | SIL Open Font License 1.1 |

Leaflet, Leaflet.draw, Chart.js и файлы шрифта Inter поставляются в
`frontend/vendor/`. Сведения о точных условиях и авторских правах смотрите в
заголовках vendor-файлов и репозиториях проектов. Для всех компонентов,
лицензия которых не указана в этой таблице или не подтверждается vendor-файлом,
см. репозиторий проекта.
