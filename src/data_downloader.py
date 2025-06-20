"""
Módulo para descargar datos CSV de internet por satélite desde fuentes públicas.
"""

import requests
import pandas as pd
import os
import logging
from typing import Dict, List, Optional
from urllib.parse import urljoin
import time

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SatelliteDataDownloader:
    """Clase para descargar datos de internet por satélite desde diferentes fuentes."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Inicializar el descargador de datos.
        
        Args:
            data_dir: Directorio donde almacenar los datos descargados
        """
        self.data_dir = data_dir
        self.create_data_directory()
        
        # URLs de fuentes de datos públicas
        self.data_sources = {
            "itu_broadband": {
                "url": "https://www.itu.int/en/ITU-D/Statistics/Documents/statistics/2023/Fixed_broadband_subscriptions_2023.xlsx",
                "description": "Suscripciones de banda ancha fija por país (ITU)"
            },
            "satellite_operators": {
                "url": "https://www.itu.int/en/ITU-R/space/snl/Documents/SNL.xlsx", 
                "description": "Lista de operadores de satélite (ITU)"
            },
            "global_connectivity": {
                "url": "https://datahub.io/core/global-temp/r/annual.csv",
                "description": "Datos de conectividad global (ejemplo)"
            }
        }
    
    def create_data_directory(self) -> None:
        """Crear directorio de datos si no existe."""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            logger.info(f"Directorio creado: {self.data_dir}")
    
    def download_csv_from_url(self, url: str, filename: str, max_retries: int = 3) -> Optional[pd.DataFrame]:
        """
        Descargar un archivo CSV desde una URL.
        
        Args:
            url: URL del archivo a descargar
            filename: Nombre del archivo local
            max_retries: Número máximo de intentos
            
        Returns:
            DataFrame con los datos o None si falla
        """
        filepath = os.path.join(self.data_dir, filename)
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Descargando datos desde: {url}")
                
                # Realizar la petición HTTP
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                
                # Guardar el archivo
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                # Intentar leer como CSV
                if filename.endswith('.csv'):
                    df = pd.read_csv(filepath)
                    logger.info(f"Descarga exitosa: {filename} ({len(df)} filas)")
                    return df
                else:
                    logger.info(f"Archivo descargado: {filename}")
                    return None
                    
            except requests.RequestException as e:
                logger.warning(f"Intento {attempt + 1} fallido para {url}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Backoff exponencial
                    
            except Exception as e:
                logger.error(f"Error procesando {filename}: {e}")
                break
        
        logger.error(f"No se pudo descargar {url} después de {max_retries} intentos")
        return None
    
    def create_sample_satellite_data(self) -> pd.DataFrame:
        """
        Crear datos de muestra de internet por satélite para testing.
        
        Returns:
            DataFrame con datos de muestra
        """
        import numpy as np
        
        # Crear datos sintéticos realistas
        countries = ['USA', 'Canada', 'Brazil', 'UK', 'Germany', 'Japan', 'Australia', 'India', 'China', 'Mexico']
        providers = ['Starlink', 'OneWeb', 'Viasat', 'HughesNet', 'Iridium']
        
        data = []
        for country in countries:
            for provider in providers:
                # Generar métricas realistas
                coverage = np.random.uniform(20, 95)  # % de cobertura
                avg_speed = np.random.uniform(25, 150)  # Mbps
                latency = np.random.uniform(20, 600)  # ms
                subscribers = np.random.randint(1000, 500000)
                monthly_cost = np.random.uniform(50, 200)  # USD
                
                data.append({
                    'country': country,
                    'provider': provider,
                    'coverage_percent': round(coverage, 2),
                    'avg_download_speed_mbps': round(avg_speed, 2),
                    'avg_latency_ms': round(latency, 2),
                    'subscribers': subscribers,
                    'monthly_cost_usd': round(monthly_cost, 2),
                    'year': 2024
                })
        
        df = pd.DataFrame(data)
        
        # Guardar datos de muestra
        sample_path = os.path.join(self.data_dir, 'satellite_internet_sample.csv')
        df.to_csv(sample_path, index=False)
        logger.info(f"Datos de muestra creados: {sample_path}")
        
        return df
    
    def download_real_data_sources(self) -> Dict[str, pd.DataFrame]:
        """
        Descargar datos de fuentes reales disponibles.
        
        Returns:
            Diccionario con los DataFrames descargados
        """
        datasets = {}
        
        # URLs de datos reales disponibles
        real_sources = {
            "global_internet_users": {
                "url": "https://raw.githubusercontent.com/datasets/internet-users/master/data/internet-users.csv",
                "filename": "global_internet_users.csv"
            },
            "broadband_coverage": {
                "url": "https://raw.githubusercontent.com/datasets/broadband-coverage/master/data/broadband-coverage.csv", 
                "filename": "broadband_coverage.csv"
            }
        }
        
        for name, source in real_sources.items():
            try:
                df = self.download_csv_from_url(source["url"], source["filename"])
                if df is not None:
                    datasets[name] = df
            except Exception as e:
                logger.warning(f"No se pudo descargar {name}: {e}")
        
        return datasets
    
    def download_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Descargar todos los datos disponibles.
        
        Returns:
            Diccionario con todos los DataFrames
        """
        logger.info("Iniciando descarga de datos de internet por satélite...")
        
        all_datasets = {}
        
        # 1. Intentar descargar datos reales
        real_data = self.download_real_data_sources()
        all_datasets.update(real_data)
        
        # 2. Crear datos de muestra para satélites
        sample_data = self.create_sample_satellite_data()
        all_datasets['satellite_sample'] = sample_data
        
        # 3. Resumen de descarga
        logger.info(f"Descarga completada. {len(all_datasets)} datasets disponibles:")
        for name, df in all_datasets.items():
            logger.info(f"  - {name}: {len(df)} filas, {len(df.columns)} columnas")
        
        return all_datasets
    
    def get_available_files(self) -> List[str]:
        """
        Obtener lista de archivos disponibles en el directorio de datos.
        
        Returns:
            Lista de nombres de archivos
        """
        if not os.path.exists(self.data_dir):
            return []
        
        files = [f for f in os.listdir(self.data_dir) if f.endswith(('.csv', '.xlsx', '.json'))]
        return files


def main():
    """Función principal para testing del módulo."""
    downloader = SatelliteDataDownloader()
    
    # Descargar todos los datos
    datasets = downloader.download_all_data()
    
    # Mostrar información de los datasets
    print("\n=== Resumen de Datos Descargados ===")
    for name, df in datasets.items():
        print(f"\n{name}:")
        print(f"  Dimensiones: {df.shape}")
        print(f"  Columnas: {list(df.columns)}")
        if len(df) > 0:
            print(f"  Muestra:")
            print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
