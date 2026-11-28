"""Pobiera sprzedaże lokali mieszkalnych z usługi WFS RCN do pliku CSV.
Użycie: python fetch_rcn.py 1465 2023-01-01 warszawa.csv
        (1465 = kod TERYT miasta na prawach powiatu, np. Warszawa)
"""
import sys
import xml.etree.ElementTree as ET
import pandas as pd
import requests

URL = "https://mapy.geoportal.gov.pl/wss/service/rcn"
NS = {"ms": "http://mapserver.gis.umn.edu/mapserver", "gml": "http://www.opengis.net/gml/3.2"}
PAGE = 2000  # ile rekordów na jedno zapytanie

teryt, date_from, out_csv = sys.argv[1], sys.argv[2], sys.argv[3]

# Filtr po stronie serwera: ten powiat ORAZ lokale mieszkalne ORAZ data >= date_from
FILTER = f"""<fes:Filter xmlns:fes="http://www.opengis.net/fes/2.0"><fes:And>
<fes:PropertyIsEqualTo><fes:ValueReference>TERYT</fes:ValueReference><fes:Literal>{teryt}</fes:Literal></fes:PropertyIsEqualTo>
<fes:PropertyIsEqualTo><fes:ValueReference>LOK_FUNKCJA</fes:ValueReference><fes:Literal>mieszkalna</fes:Literal></fes:PropertyIsEqualTo>
<fes:PropertyIsGreaterThanOrEqualTo><fes:ValueReference>DOK_DATA</fes:ValueReference><fes:Literal>{date_from}</fes:Literal></fes:PropertyIsGreaterThanOrEqualTo>
</fes:And></fes:Filter>"""

BASE = {"SERVICE": "WFS", "VERSION": "2.0.0", "REQUEST": "GetFeature",
        "TYPENAMES": "ms:lokale", "FILTER": FILTER}

# 1) pytamy serwer, ile rekordów pasuje
hits = ET.fromstring(requests.get(URL, params={**BASE, "RESULTTYPE": "hits"}, timeout=600).content)
total = int(hits.get("numberMatched"))
print("rows to download:", total)

# 2) pobieramy strona po stronie
#    Uwaga: serwer po cichu pomija część rekordów (~10%), więc strony bywają krótsze albo puste.
rows = []
for start in range(0, total, PAGE):
    root = ET.fromstring(requests.get(URL, params={**BASE, "COUNT": PAGE, "STARTINDEX": start}, timeout=600).content)
    feats = root.findall(".//ms:lokale", NS)
    for f in feats:
        # każde pole XML -> kolumna (bez geometrii)
        row = {el.tag.split("}")[1]: (el.text or "").strip() for el in f if el.tag != f"{{{NS['ms']}}}GEOMETRIA"}
        row["DOK_DATA"] = f.findtext(".//ms:DOK_DATA//gml:timePosition", "", NS)  # data aktu
        pos = f.findtext(".//gml:pos", "", NS).split()
        row["y_2180"], row["x_2180"] = (pos + ["", ""])[:2]  # współrzędne EPSG:2180 w metrach
        rows.append(row)
    print(f"got {len(rows)} of {total} rows")

pd.DataFrame(rows).drop(columns=["TRAN_WERSJA_ID"], errors="ignore").to_csv(out_csv, index=False)
print("saved", out_csv)