#!/usr/bin/env python3
"""
script para probar diferentes rangos de fechas fácilmente
"""

import subprocess
import sys

def run_model_with_dates(year_from, month_from, year_to, month_to):
    """ejecuta el modelo con fechas específicas"""
    print(f"\n{'='*60}")
    print(f"probando modelo con datos desde {year_from}-{month_from:02d} hasta {year_to}-{month_to:02d}")
    print(f"{'='*60}")
    
    # crear archivo temporal con las fechas
    with open('recent_data_model.py', 'r') as f:
        content = f.read()
    
    # reemplazar las fechas
    content = content.replace('YEAR_FROM = 2020', f'YEAR_FROM = {year_from}')
    content = content.replace('MONTH_FROM = 1', f'MONTH_FROM = {month_from}')
    content = content.replace('YEAR_TO = 2024', f'YEAR_TO = {year_to}')
    content = content.replace('MONTH_TO = 12', f'MONTH_TO = {month_to}')
    
    # escribir archivo temporal
    with open('temp_model.py', 'w') as f:
        f.write(content)
    
    # ejecutar modelo
    try:
        result = subprocess.run([sys.executable, 'temp_model.py'], 
                              capture_output=True, text=True, timeout=300)
        print(result.stdout)
        if result.stderr:
            print("errores:", result.stderr)
    except subprocess.TimeoutExpired:
        print("timeout - modelo tardó más de 5 minutos")
    except Exception as e:
        print(f"error ejecutando modelo: {e}")
    
    # limpiar archivo temporal
    import os
    if os.path.exists('temp_model.py'):
        os.remove('temp_model.py')

def main():
    """función principal para probar diferentes rangos"""
    
    # configuraciones a probar
    configs = [
        # (year_from, month_from, year_to, month_to, descripción)
        (2024, 1, 2024, 12, "solo 2024"),
        (2023, 1, 2024, 12, "2023-2024"),
        (2022, 1, 2024, 12, "2022-2024"),
        (2024, 7, 2024, 12, "solo últimos 6 meses de 2024"),
        (2024, 10, 2024, 12, "solo últimos 3 meses de 2024"),
    ]
    
    print("probando diferentes rangos de fechas...")
    print("esto puede tomar varios minutos...")
    
    for year_from, month_from, year_to, month_to, desc in configs:
        print(f"\nprobando: {desc}")
        run_model_with_dates(year_from, month_from, year_to, month_to)
        
        # pausa entre ejecuciones
        input("\npresiona enter para continuar con el siguiente rango...")
    
    print("\n¡todas las pruebas completadas!")
    print("revisa los archivos submission_*.csv generados")

if __name__ == "__main__":
    main()
