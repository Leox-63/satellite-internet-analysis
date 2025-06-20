"""
Módulo para análisis de datos de internet por satélite.
"""

import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, List, Tuple, Optional

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SatelliteDataAnalyzer:
    """Clase para analizar datos de internet por satélite."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Inicializar el analizador de datos.
        
        Args:
            data_dir: Directorio donde están almacenados los datos
        """
        self.data_dir = data_dir
        self.datasets = {}
        self.load_available_data()
    
    def load_available_data(self) -> None:
        """Cargar todos los archivos CSV disponibles."""
        if not os.path.exists(self.data_dir):
            logger.warning(f"Directorio de datos no encontrado: {self.data_dir}")
            return
        
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        
        for file in csv_files:
            try:
                filepath = os.path.join(self.data_dir, file)
                dataset_name = file.replace('.csv', '')
                self.datasets[dataset_name] = pd.read_csv(filepath)
                logger.info(f"Dataset cargado: {dataset_name} ({len(self.datasets[dataset_name])} filas)")
            except Exception as e:
                logger.error(f"Error cargando {file}: {e}")
    
    def get_basic_statistics(self, dataset_name: Optional[str] = None) -> Dict:
        """
        Obtener estadísticas básicas de los datos.
        
        Args:
            dataset_name: Nombre específico del dataset (opcional)
            
        Returns:
            Diccionario con estadísticas
        """
        stats = {}
        
        if dataset_name and dataset_name in self.datasets:
            datasets_to_analyze = {dataset_name: self.datasets[dataset_name]}
        else:
            datasets_to_analyze = self.datasets
        
        for name, df in datasets_to_analyze.items():
            dataset_stats = {
                'total_records': len(df),
                'columns': list(df.columns),
                'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
                'categorical_columns': list(df.select_dtypes(include=['object']).columns),
                'missing_values': df.isnull().sum().to_dict(),
                'data_types': df.dtypes.to_dict()
            }
            
            # Estadísticas específicas para datos numéricos
            numeric_stats = {}
            for col in df.select_dtypes(include=[np.number]).columns:
                numeric_stats[col] = {
                    'mean': df[col].mean(),
                    'median': df[col].median(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'quartiles': {
                        'Q1': df[col].quantile(0.25),
                        'Q3': df[col].quantile(0.75)
                    }
                }
            
            dataset_stats['numeric_statistics'] = numeric_stats
            stats[name] = dataset_stats
        
        return stats
    
    def analyze_satellite_coverage(self) -> Dict:
        """
        Analizar datos de cobertura satelital.
        
        Returns:
            Análisis de cobertura por país y proveedor
        """
        analysis = {}
        
        # Buscar dataset de satélites
        satellite_data = None
        for name, df in self.datasets.items():
            if 'satellite' in name.lower() or 'coverage' in df.columns:
                satellite_data = df
                break
        
        if satellite_data is None:
            logger.warning("No se encontraron datos de satélites para analizar")
            return analysis
        
        # Análisis por país
        if 'country' in satellite_data.columns:
            country_analysis = satellite_data.groupby('country').agg({
                'coverage_percent': ['mean', 'max', 'min'] if 'coverage_percent' in satellite_data.columns else 'count',
                'avg_download_speed_mbps': ['mean', 'max'] if 'avg_download_speed_mbps' in satellite_data.columns else 'count',
                'subscribers': 'sum' if 'subscribers' in satellite_data.columns else 'count'
            }).round(2)
            
            analysis['by_country'] = country_analysis.to_dict()
        
        # Análisis por proveedor
        if 'provider' in satellite_data.columns:
            provider_analysis = satellite_data.groupby('provider').agg({
                'coverage_percent': ['mean', 'count'] if 'coverage_percent' in satellite_data.columns else 'count',
                'avg_download_speed_mbps': 'mean' if 'avg_download_speed_mbps' in satellite_data.columns else 'count',
                'subscribers': 'sum' if 'subscribers' in satellite_data.columns else 'count',
                'monthly_cost_usd': 'mean' if 'monthly_cost_usd' in satellite_data.columns else 'count'
            }).round(2)
            
            analysis['by_provider'] = provider_analysis.to_dict()
        
        # Top países por cobertura
        if 'coverage_percent' in satellite_data.columns and 'country' in satellite_data.columns:
            top_coverage = satellite_data.groupby('country')['coverage_percent'].mean().sort_values(ascending=False).head(10)
            analysis['top_coverage_countries'] = top_coverage.to_dict()
        
        # Correlaciones
        numeric_cols = satellite_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            correlations = satellite_data[numeric_cols].corr()
            analysis['correlations'] = correlations.to_dict()
        
        return analysis
    
    def analyze_speed_trends(self) -> Dict:
        """
        Analizar tendencias de velocidad de internet satelital.
        
        Returns:
            Análisis de tendencias de velocidad
        """
        analysis = {}
        
        for name, df in self.datasets.items():
            if 'speed' in df.columns or any('mbps' in col.lower() for col in df.columns):
                speed_cols = [col for col in df.columns if 'speed' in col.lower() or 'mbps' in col.lower()]
                
                for col in speed_cols:
                    if df[col].dtype in [np.number]:
                        analysis[f'{name}_{col}'] = {
                            'average_speed': df[col].mean(),
                            'median_speed': df[col].median(),
                            'speed_distribution': {
                                'below_25mbps': (df[col] < 25).sum(),
                                '25_100mbps': ((df[col] >= 25) & (df[col] < 100)).sum(),
                                'above_100mbps': (df[col] >= 100).sum()
                            }
                        }
        
        return analysis
    
    def compare_providers(self) -> Dict:
        """
        Comparar diferentes proveedores de internet satelital.
        
        Returns:
            Comparación entre proveedores
        """
        comparison = {}
        
        # Buscar datos con información de proveedores
        provider_data = None
        for name, df in self.datasets.items():
            if 'provider' in df.columns:
                provider_data = df
                break
        
        if provider_data is None:
            return comparison
        
        providers = provider_data['provider'].unique()
        
        for provider in providers:
            provider_subset = provider_data[provider_data['provider'] == provider]
            
            provider_metrics = {
                'countries_served': provider_subset['country'].nunique() if 'country' in provider_subset.columns else 0,
                'total_subscribers': provider_subset['subscribers'].sum() if 'subscribers' in provider_subset.columns else 0,
                'avg_coverage': provider_subset['coverage_percent'].mean() if 'coverage_percent' in provider_subset.columns else 0,
                'avg_speed': provider_subset['avg_download_speed_mbps'].mean() if 'avg_download_speed_mbps' in provider_subset.columns else 0,
                'avg_cost': provider_subset['monthly_cost_usd'].mean() if 'monthly_cost_usd' in provider_subset.columns else 0,
                'avg_latency': provider_subset['avg_latency_ms'].mean() if 'avg_latency_ms' in provider_subset.columns else 0
            }
            
            comparison[provider] = {k: round(v, 2) if isinstance(v, float) else v for k, v in provider_metrics.items()}
        
        return comparison
    
    def generate_summary_report(self) -> str:
        """
        Generar un reporte resumen del análisis.
        
        Returns:
            Reporte en formato texto
        """
        report = []
        report.append("=== REPORTE DE ANÁLISIS DE INTERNET SATELITAL ===\n")
        
        # Información general
        report.append(f"Datasets disponibles: {len(self.datasets)}")
        total_records = sum(len(df) for df in self.datasets.values())
        report.append(f"Total de registros: {total_records}\n")
        
        # Estadísticas básicas
        stats = self.get_basic_statistics()
        for dataset_name, dataset_stats in stats.items():
            report.append(f"Dataset: {dataset_name}")
            report.append(f"  - Registros: {dataset_stats['total_records']}")
            report.append(f"  - Columnas: {len(dataset_stats['columns'])}")
            report.append(f"  - Columnas numéricas: {len(dataset_stats['numeric_columns'])}")
            
            # Valores faltantes
            missing_total = sum(dataset_stats['missing_values'].values())
            if missing_total > 0:
                report.append(f"  - Valores faltantes: {missing_total}")
            report.append("")
        
        # Análisis de cobertura
        coverage_analysis = self.analyze_satellite_coverage()
        if coverage_analysis:
            report.append("=== ANÁLISIS DE COBERTURA ===")
            
            if 'top_coverage_countries' in coverage_analysis:
                report.append("Top 5 países por cobertura:")
                top_countries = list(coverage_analysis['top_coverage_countries'].items())[:5]
                for country, coverage in top_countries:
                    report.append(f"  - {country}: {coverage:.1f}%")
                report.append("")
        
        # Comparación de proveedores
        provider_comparison = self.compare_providers()
        if provider_comparison:
            report.append("=== COMPARACIÓN DE PROVEEDORES ===")
            for provider, metrics in provider_comparison.items():
                report.append(f"{provider}:")
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)) and value > 0:
                        report.append(f"  - {metric}: {value}")
                report.append("")
        
        return "\n".join(report)
    
    def export_analysis_results(self, output_dir: str = "analysis_results") -> None:
        """
        Exportar resultados del análisis a archivos.
        
        Args:
            output_dir: Directorio donde guardar los resultados
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Exportar estadísticas básicas
        stats = self.get_basic_statistics()
        stats_df = pd.DataFrame([(dataset, stat_name, stat_value) 
                                for dataset, dataset_stats in stats.items() 
                                for stat_name, stat_value in dataset_stats.items()
                                if isinstance(stat_value, (int, float, str))])
        stats_df.columns = ['Dataset', 'Statistic', 'Value']
        stats_df.to_csv(os.path.join(output_dir, 'basic_statistics.csv'), index=False)
        
        # Exportar reporte resumen
        report = self.generate_summary_report()
        with open(os.path.join(output_dir, 'summary_report.txt'), 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Resultados exportados a: {output_dir}")


def main():
    """Función principal para testing del módulo."""
    analyzer = SatelliteDataAnalyzer()
    
    # Mostrar estadísticas básicas
    print("=== ESTADÍSTICAS BÁSICAS ===")
    stats = analyzer.get_basic_statistics()
    for name, stat in stats.items():
        print(f"\n{name}: {stat['total_records']} registros, {len(stat['columns'])} columnas")
    
    # Generar reporte
    print("\n=== REPORTE RESUMEN ===")
    report = analyzer.generate_summary_report()
    print(report)


if __name__ == "__main__":
    main()
