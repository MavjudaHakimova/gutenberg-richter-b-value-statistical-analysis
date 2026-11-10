"""Загрузка событий и магнитуд из ISC Bulletin QuakeML.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
import pandas as pd
import requests

BASE_URL = 'https://www.isc.ac.uk/cgi-bin/web-db-run'
NS = {'bed': 'http://quakeml.org/xmlns/bed/1.2'}


def _value(node, path: str):
    """Безопасно получить текст необязательного поля QuakeML."""
    sub = node.find(path, NS)
    return sub.text if sub is not None else None


def get_quake_ml(url: str) -> str:
    """Получить XML с контролем кода ответа и ограничением ожидания."""
    response = requests.get(url, timeout=75, headers={'User-Agent':'GR-paper-repro/1.0'})
    response.raise_for_status()
    return response.text


def extract_events(xml_text: str) -> pd.DataFrame:
    """Прочитать координаты, время и тип сейсмического события."""
    root = ET.fromstring(xml_text)
    rows = []
    for event in root.findall('.//bed:event', NS):
        origin = event.find('bed:origin', NS)
        rows.append({
            'event_id': (event.get('publicID') or '').replace('smi:ISC/evid=', ''),
            'event_type': _value(event,'bed:type'),
            'event_type_certainty': _value(event,'bed:typeCertainty'),
            'event_dttm': _value(origin,'.//bed:time/bed:value') if origin is not None else None,
            'lat': _value(origin,'.//bed:latitude/bed:value') if origin is not None else None,
            'lon': _value(origin,'.//bed:longitude/bed:value') if origin is not None else None,
            'depth': _value(origin,'.//bed:depth/bed:value') if origin is not None else None,
        })
    data=pd.DataFrame(rows)
    for column in ('lat','lon','depth'):
        if column in data:
            data[column]=pd.to_numeric(data[column],errors='coerce')
    return data


def extract_magnitudes(xml_text: str) -> pd.DataFrame:
    """Считать исходные магнитуды и их шкалы (mb, MS, ML, Mw и др.)."""
    root=ET.fromstring(xml_text)
    rows=[]
    for event in root.findall('.//bed:event',NS):
        event_id=(event.get('publicID') or '').replace('smi:ISC/evid=','')
        for mag in event.findall('bed:magnitude',NS):
            rows.append({
                'event_id':event_id,
                'magnitude_id':(mag.get('publicID') or '').replace('smi:ISC/magid=',''),
                'mag_value':_value(mag,'bed:mag/bed:value'),
                'mag_type':_value(mag,'bed:type'),
                'author_nm':_value(mag,'bed:creationInfo/bed:author'),
            })
    data=pd.DataFrame(rows)
    if 'mag_value' in data:
        data['mag_value']=pd.to_numeric(data['mag_value'],errors='coerce')
    return data


def get_events_and_mags(start_dt: str, end_dt: str, min_mag: float = 3.5):
    """Получить два DataFrame за указанный интервал YYYY-MM-DD.

    ISC ограничивает размер ответов: для больших периодов запрашивайте
    данные частями, например по месяцам. Магнитуды здесь ещё не приведены к Mw.
    """
    first=datetime.strptime(start_dt,'%Y-%m-%d')
    last=datetime.strptime(end_dt,'%Y-%m-%d')
    if last < first:
        raise ValueError('Конец интервала не может быть раньше начала')
    params={
        'request':'COMPREHENSIVE','out_format':'CATQuakeML',
        'searchshape':'RECT','start_year':first.year,
        'start_month':first.month,'start_day':first.day,
        'start_time':'00:00:00','end_year':last.year,
        'end_month':last.month,'end_day':last.day,
        'end_time':'23:59:59','min_mag':min_mag,
    }
    response=requests.get(BASE_URL,params=params,timeout=90)
    response.raise_for_status()
    return extract_events(response.text),extract_magnitudes(response.text)


def save_interval(start_dt: str, end_dt: str, output_dir: str | Path='downloads',
                 min_mag: float = 3.5) -> None:
    """Сохранить скачанные события и магнитуды в отдельные CSV-файлы."""
    events,mags=get_events_and_mags(start_dt,end_dt,min_mag)
    folder=Path(output_dir)
    folder.mkdir(parents=True,exist_ok=True)
    events.to_csv(folder/'events.csv',index=False)
    mags.to_csv(folder/'magnitudes.csv',index=False)
    print(f'Сохранено событий: {len(events)}, магнитуд: {len(mags)}')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Загрузка небольшого интервала из каталога ISC')
    parser.add_argument('--start',required=True,help='Начало: ГГГГ-ММ-ДД')
    parser.add_argument('--end',required=True,help='Конец: ГГГГ-ММ-ДД')
    parser.add_argument('--min-mag',type=float,default=3.5)
    parser.add_argument('--out',default='downloads')
    args=parser.parse_args()
    save_interval(args.start,args.end,args.out,args.min_mag)
