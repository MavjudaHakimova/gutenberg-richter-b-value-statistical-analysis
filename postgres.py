from __future__ import annotations
import os


def make_engine():
    """Создать соединение, только когда оно действительно необходимо."""
    from sqlalchemy import create_engine
    url=os.getenv('EARTHQUAKE_DB_URL')
    if not url:
        raise RuntimeError('Не задан EARTHQUAKE_DB_URL. См. .env.example')
    return create_engine(url,pool_pre_ping=True)


def load_to_postgres(df, table_name: str, schema: str='earthquakes_gr_low'):
    """Добавить таблицу pandas в существующую схему PostgreSQL."""
    engine=make_engine()
    try:
        df.to_sql(table_name,engine,schema=schema,if_exists='append',index=False)
    finally:
        engine.dispose()


def load_to_postgis(df, table_name: str, schema: str='earthquakes_gr_low'):
    """Добавить GeoDataFrame с геометрией в существующую таблицу PostGIS."""
    engine=make_engine()
    try:
        df.to_postgis(table_name,engine,schema=schema,if_exists='append',index=False)
    finally:
        engine.dispose()
