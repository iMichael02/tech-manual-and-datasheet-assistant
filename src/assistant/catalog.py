"""Curated catalog of the reference documents.

Maps each PDF to the part names users are likely to type. Aliases matter because
the files are named inconsistently (e.g. ``1893_datasheet.pdf`` is the MPL3115A2).
"""
from __future__ import annotations

# filename -> (display title, [aliases])
CATALOG: dict[str, tuple[str, list[str]]] = {
    "1893_datasheet.pdf": ("MPL3115A2 I2C precision altimeter / pressure sensor", ["MPL3115A2", "MPL3115"]),
    "1899_HTU21D.pdf": ("HTU21D(F) digital humidity & temperature sensor", ["HTU21D", "HTU21DF", "HTU21"]),
    "BST-BMP180-DS000-09.pdf": ("BMP180 digital pressure sensor", ["BMP180"]),
    "BST-BMP280-DS001-11.pdf": ("BMP280 digital pressure sensor", ["BMP280"]),
    "Data+Sheet+DS1000E.pdf": ("Rigol DS1000E series oscilloscope data sheet", ["DS1000E", "DS1052E", "DS1102E", "RIGOL"]),
    "Users Guide DS1000E.pdf": ("Rigol DS1000E series oscilloscope user's guide", ["DS1000E", "DS1052E", "DS1102E", "RIGOL"]),
    # NB: despite its name, DHT22.pdf is an OLDER AM2303 datasheet (3.3-6V, 20 m) that disagrees with the
    # newer AM2303 PDF (3.3-5.5V, 100 m). Kept as a deliberate conflicting-sources test case.
    "DHT22.pdf": ("AM2303 humidity & temperature sensor (older revision, file named DHT22.pdf)", ["AM2303"]),
    "Digital+humidity+and+temperature+sensor+AM2302.pdf": ("AM2302 digital humidity & temperature sensor", ["AM2302", "DHT22"]),
    "Digital+humidity+and+temperature+sensor+AM2303.pdf": ("AM2303 digital humidity & temperature sensor", ["AM2303"]),
    "DS18B20.pdf": ("DS18B20 programmable-resolution 1-Wire digital thermometer", ["DS18B20", "DS18S20", "DS1822"]),
    "hdc1008.pdf": ("HDC1008 low-power digital humidity & temperature sensor", ["HDC1008"]),
    "MAX31855.pdf": ("MAX31855 cold-junction compensated thermocouple-to-digital converter", ["MAX31855"]),
    "SLHT5.pdf": ("SLHT5 soil temperature & humidity sensor", ["SLHT5"]),
    "tmp006ug.pdf": ("TMP006 infrared thermopile sensor user's guide", ["TMP006", "TMP007"]),
}

# Alphanumeric tokens that look like part numbers but are generic technology names.
NOT_PARTS = {"RS232", "RS485", "RS422", "USB20", "IEEE1394", "CMOS12", "TTL5"}
