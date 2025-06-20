"""
Módulo para crear visualizaciones de datos de internet por satélite.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import logging
from typing import Dict, List, Optional, Tuple

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configurar estilo de matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class SatelliteDataVisualizer:
    """Clase para crear visualizaciones de datos de internet por satélite."""
    
    def __init__(self, data_dir: str = "data", output_dir: str = "visualizations"):
        """
        Inicializar el visualizador de datos.
        
        Args:
            data_dir: Directorio donde están los datos
            output_dir: Directorio donde guardar las visualizaciones
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.datasets = {}
        self.create_output_directory()
        self.load_data()
    
    def create_output_directory(self) -> None:
        """Crear directorio de salida si no existe."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Directorio de visualizaciones creado: {self.output_dir}")
    
    def load_data(self) -> None:
        """Cargar datos disponibles."""
        if not os.path.exists(self.data_dir):
            logger.warning(f"Directorio de datos no encontrado: {self.data_dir}")
            return
        
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        for file in csv_files:
            try:
                filepath = os.path.join(self.data_dir, file)
                dataset_name = file.replace('.csv', '')
                self.datasets[dataset_name] = pd.read_csv(filepath)
                logger.info(f"Dataset cargado para visualización: {dataset_name}")
            except Exception as e:
                logger.error(f"Error cargando {file}: {e}")
    
    def create_coverage_map(self, save_plot: bool = True) -> go.Figure:
        """
        Crear mapa de cobertura satelital por país.
        
        Args:
            save_plot: Si guardar el gráfico como archivo
            
        Returns:
            Figura de Plotly
        """
        # Buscar datos con información geográfica
        geo_data = None
        for name, df in self.datasets.items():
            if 'country' in df.columns and any('coverage' in col.lower() for col in df.columns):
                geo_data = df
                break
        
        if geo_data is None:
            logger.warning("No se encontraron datos geográficos para el mapa")
            return None
        
        # Encontrar columna de cobertura
        coverage_col = None
        for col in geo_data.columns:
            if 'coverage' in col.lower():
                coverage_col = col
                break
        
        if coverage_col is None:
            logger.warning("No se encontró columna de cobertura")
            return None
        
        # Agregar datos por país
        country_coverage = geo_data.groupby('country')[coverage_col].mean().reset_index()
        
        # Crear mapa
        fig = px.choropleth(
            country_coverage,
            locations='country',
            color=coverage_col,
            locationmode='country names',
            title='Cobertura de Internet Satelital por País',
            color_continuous_scale='Viridis',
            labels={coverage_col: 'Cobertura (%)'}
        )
        
        fig.update_layout(
            title_font_size=20,
            title_x=0.5,
            geo=dict(showframe=False, showcoastlines=True)
        )
        
        if save_plot:
            filepath = os.path.join(self.output_dir, 'coverage_map.html')
            fig.write_html(filepath)
            logger.info(f"Mapa de cobertura guardado: {filepath}")
        
        return fig
    
    def create_speed_comparison(self, save_plot: bool = True) -> plt.Figure:
        """
        Crear gráfico de comparación de velocidades por proveedor.
        
        Args:
            save_plot: Si guardar el gráfico como archivo
            
        Returns:
            Figura de matplotlib
        """
        # Buscar datos con velocidades y proveedores
        speed_data = None
        speed_col = None
        
        for name, df in self.datasets.items():
            if 'provider' in df.columns:
                for col in df.columns:
                    if 'speed' in col.lower() or 'mbps' in col.lower():
                        speed_data = df
                        speed_col = col
                        break
                if speed_data is not None:
                    break
        
        if speed_data is None or speed_col is None:
            logger.warning("No se encontraron datos de velocidad por proveedor")
            return None
        
        # Crear gráfico
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Box plot de velocidades por proveedor
        sns.boxplot(data=speed_data, x='provider', y=speed_col, ax=ax1)
        ax1.set_title('Distribución de Velocidades por Proveedor')
        ax1.set_xlabel('Proveedor')
        ax1.set_ylabel('Velocidad (Mbps)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Gráfico de barras con velocidad promedio
        avg_speeds = speed_data.groupby('provider')[speed_col].mean().sort_values(ascending=False)
        avg_speeds.plot(kind='bar', ax=ax2, color='skyblue')
        ax2.set_title('Velocidad Promedio por Proveedor')
        ax2.set_xlabel('Proveedor')
        ax2.set_ylabel('Velocidad Promedio (Mbps)')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_plot:
            filepath = os.path.join(self.output_dir, 'speed_comparison.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico de velocidades guardado: {filepath}")
        
        return fig
    
    def create_cost_analysis(self, save_plot: bool = True) -> go.Figure:
        """
        Crear análisis de costos de internet satelital.
        
        Args:
            save_plot: Si guardar el gráfico como archivo
            
        Returns:
            Figura de Plotly
        """
        # Buscar datos con costos
        cost_data = None
        cost_col = None
        
        for name, df in self.datasets.items():
            for col in df.columns:
                if 'cost' in col.lower() or 'price' in col.lower():
                    cost_data = df
                    cost_col = col
                    break
            if cost_data is not None:
                break
        
        if cost_data is None:
            logger.warning("No se encontraron datos de costos")
            return None
        
        # Crear subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Distribución de Costos', 'Costo vs Velocidad', 
                          'Costo por País', 'Costo por Proveedor'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Histograma de costos
        fig.add_trace(
            go.Histogram(x=cost_data[cost_col], name='Distribución de Costos',
                        nbinsx=20, marker_color='lightblue'),
            row=1, col=1
        )
        
        # 2. Scatter plot costo vs velocidad (si existe)
        speed_col = None
        for col in cost_data.columns:
            if 'speed' in col.lower() or 'mbps' in col.lower():
                speed_col = col
                break
        
        if speed_col is not None:
            fig.add_trace(
                go.Scatter(x=cost_data[cost_col], y=cost_data[speed_col],
                          mode='markers', name='Costo vs Velocidad',
                          marker=dict(color='orange', size=8)),
                row=1, col=2
            )
        
        # 3. Costo promedio por país (si existe)
        if 'country' in cost_data.columns:
            country_costs = cost_data.groupby('country')[cost_col].mean().sort_values(ascending=False).head(10)
            fig.add_trace(
                go.Bar(x=list(country_costs.index), y=list(country_costs.values),
                      name='Costo por País', marker_color='green'),
                row=2, col=1
            )
        
        # 4. Costo promedio por proveedor (si existe)
        if 'provider' in cost_data.columns:
            provider_costs = cost_data.groupby('provider')[cost_col].mean().sort_values(ascending=False)
            fig.add_trace(
                go.Bar(x=list(provider_costs.index), y=list(provider_costs.values),
                      name='Costo por Proveedor', marker_color='red'),
                row=2, col=2
            )
        
        # Actualizar layout
        fig.update_layout(
            title_text="Análisis de Costos de Internet Satelital",
            title_x=0.5,
            height=800,
            showlegend=False
        )
        
        if save_plot:
            filepath = os.path.join(self.output_dir, 'cost_analysis.html')
            fig.write_html(filepath)
            logger.info(f"Análisis de costos guardado: {filepath}")
        
        return fig
    
    def create_latency_analysis(self, save_plot: bool = True) -> plt.Figure:
        """
        Crear análisis de latencia por proveedor y región.
        
        Args:
            save_plot: Si guardar el gráfico como archivo
            
        Returns:
            Figura de matplotlib
        """
        # Buscar datos con latencia
        latency_data = None
        latency_col = None
        
        for name, df in self.datasets.items():
            for col in df.columns:
                if 'latency' in col.lower() or 'ping' in col.lower():
                    latency_data = df
                    latency_col = col
                    break
            if latency_data is not None:
                break
        
        if latency_data is None:
            logger.warning("No se encontraron datos de latencia")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Distribución de latencia
        axes[0,0].hist(latency_data[latency_col], bins=20, color='lightcoral', alpha=0.7)
        axes[0,0].set_title('Distribución de Latencia')
        axes[0,0].set_xlabel('Latencia (ms)')
        axes[0,0].set_ylabel('Frecuencia')
        
        # 2. Latencia por proveedor (si existe)
        if 'provider' in latency_data.columns:
            provider_latency = latency_data.groupby('provider')[latency_col].mean().sort_values()
            provider_latency.plot(kind='barh', ax=axes[0,1], color='lightgreen')
            axes[0,1].set_title('Latencia Promedio por Proveedor')
            axes[0,1].set_xlabel('Latencia (ms)')
        
        # 3. Box plot de latencia por proveedor
        if 'provider' in latency_data.columns:
            sns.boxplot(data=latency_data, x='provider', y=latency_col, ax=axes[1,0])
            axes[1,0].set_title('Distribución de Latencia por Proveedor')
            axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. Latencia vs velocidad (si existe velocidad)
        speed_col = None
        for col in latency_data.columns:
            if 'speed' in col.lower() or 'mbps' in col.lower():
                speed_col = col
                break
        
        if speed_col is not None:
            axes[1,1].scatter(latency_data[latency_col], latency_data[speed_col], 
                            alpha=0.6, color='purple')
            axes[1,1].set_title('Latencia vs Velocidad')
            axes[1,1].set_xlabel('Latencia (ms)')
            axes[1,1].set_ylabel('Velocidad (Mbps)')
        
        plt.tight_layout()
        
        if save_plot:
            filepath = os.path.join(self.output_dir, 'latency_analysis.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Análisis de latencia guardado: {filepath}")
        
        return fig
    
    def create_dashboard_summary(self, save_plot: bool = True) -> go.Figure:
        """
        Crear un dashboard resumen con múltiples métricas.
        
        Args:
            save_plot: Si guardar el gráfico como archivo
            
        Returns:
            Figura de Plotly
        """
        # Buscar el dataset principal
        main_data = None
        for name, df in self.datasets.items():
            if 'satellite' in name.lower() or len(df.columns) > 5:
                main_data = df
                break
        
        if main_data is None:
            logger.warning("No se encontraron datos principales para el dashboard")
            return None
        
        # Crear dashboard con 4 subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Métricas por País', 'Distribución de Velocidades',
                          'Suscriptores por Proveedor', 'Correlación de Métricas'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Métricas por país (si existe)
        if 'country' in main_data.columns:
            countries = main_data['country'].value_counts().head(10)
            fig.add_trace(
                go.Bar(x=list(countries.index), y=list(countries.values),
                      name='Países', marker_color='lightblue'),
                row=1, col=1
            )
        
        # 2. Distribución de velocidades (si existe)
        speed_col = None
        for col in main_data.columns:
            if 'speed' in col.lower() or 'mbps' in col.lower():
                speed_col = col
                break
        
        if speed_col is not None:
            fig.add_trace(
                go.Histogram(x=main_data[speed_col], name='Velocidades',
                           marker_color='orange', nbinsx=15),
                row=1, col=2
            )
        
        # 3. Suscriptores por proveedor (si existe)
        if 'provider' in main_data.columns and 'subscribers' in main_data.columns:
            provider_subs = main_data.groupby('provider')['subscribers'].sum().sort_values(ascending=False)
            fig.add_trace(
                go.Bar(x=list(provider_subs.index), y=list(provider_subs.values),
                      name='Suscriptores', marker_color='green'),
                row=2, col=1
            )
        
        # 4. Matriz de correlación (solo columnas numéricas)
        numeric_cols = main_data.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) > 1:
            corr_matrix = main_data[numeric_cols].corr()
            fig.add_trace(
                go.Heatmap(z=corr_matrix.values,
                          x=list(corr_matrix.columns),
                          y=list(corr_matrix.columns),
                          colorscale='RdBu',
                          name='Correlaciones'),
                row=2, col=2
            )
        
        # Actualizar layout
        fig.update_layout(
            title_text="Dashboard - Análisis de Internet Satelital",
            title_x=0.5,
            height=800,
            showlegend=False
        )
        
        if save_plot:
            filepath = os.path.join(self.output_dir, 'dashboard_summary.html')
            fig.write_html(filepath)
            logger.info(f"Dashboard guardado: {filepath}")
        
        return fig
    
    def generate_all_visualizations(self) -> None:
        """Generar todas las visualizaciones disponibles."""
        logger.info("Generando todas las visualizaciones...")
        
        visualizations = [
            ('Mapa de Cobertura', self.create_coverage_map),
            ('Comparación de Velocidades', self.create_speed_comparison),
            ('Análisis de Costos', self.create_cost_analysis),
            ('Análisis de Latencia', self.create_latency_analysis),
            ('Dashboard Resumen', self.create_dashboard_summary)
        ]
        
        successful = 0
        for name, func in visualizations:
            try:
                result = func(save_plot=True)
                if result is not None:
                    successful += 1
                    logger.info(f"✓ {name} generado exitosamente")
                else:
                    logger.warning(f"✗ {name} no se pudo generar (datos insuficientes)")
            except Exception as e:
                logger.error(f"✗ Error generando {name}: {e}")
        
        logger.info(f"Visualizaciones completadas: {successful}/{len(visualizations)}")
        logger.info(f"Archivos guardados en: {self.output_dir}")


def main():
    """Función principal para testing del módulo."""
    visualizer = SatelliteDataVisualizer()
    
    print("Generando todas las visualizaciones...")
    visualizer.generate_all_visualizations()
    
    print(f"\nVisualizaciones guardadas en: {visualizer.output_dir}")
    print("Archivos HTML se pueden abrir en el navegador")
    print("Archivos PNG se pueden ver con cualquier visor de imágenes")


if __name__ == "__main__":
    main()
