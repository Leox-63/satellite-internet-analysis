<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# Instrucciones del Proyecto - Análisis de Internet por Satélite

Este es un proyecto de análisis de datos de internet por satélite con Python. Por favor:

## Estructura Modular
- Mantén la separación entre descarga de datos (`data_downloader.py`), análisis (`data_analyzer.py`) y visualización (`visualizer.py`)
- Los datos CSV se almacenan en la carpeta `data/`
- Los notebooks están en `notebooks/`

## Estilo de Código
- Usa pandas para manipulación de datos
- Matplotlib/Seaborn/Plotly para visualizaciones
- Incluye docstrings en todas las funciones
- Maneja errores apropiadamente al descargar datos
- Usa logging para reportar el progreso

## Datos de Internet por Satélite
- Enfócate en métricas como velocidad, latencia, cobertura
- Incluye datos geográficos cuando sea posible
- Considera diferentes proveedores de satélite (Starlink, OneWeb, etc.)
- Analiza tendencias temporales en la conectividad
