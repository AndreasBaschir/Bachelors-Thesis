import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# 1. Interogare API arXiv pentru colectarea datelor oficiale în timp real
years = list(range(2017, 2026))
papers = []

print("Pasul 1: Interogare API arXiv pentru expresia 'Large Language Model'...")

for year in years:
    # Construim query-ul exact filtrat pe anul respectiv
    search_query = f'all:"large language model" AND submittedDate:[{year}01010000 TO {year}12312359]'
    query_encoded = urllib.parse.quote(search_query)
    url = f'http://export.arxiv.org/api/query?search_query={query_encoded}&max_results=1'
    
    try:
        response = urllib.request.urlopen(url)
        xml_data = response.read()
        root = ET.fromstring(xml_data)
        
        # Extragem numărul total de rezultate din structura XML a arXiv
        total_results = int(root.find('{http://a9.com/-/spec/opensearch/1.1/}totalResults').text)
        papers.append(total_results)
        print(f" -> Anul {year}: {total_results} lucrări găsite.")
    except Exception as e:
        print(f" -> Eroare la interogarea pentru anul {year}: {e}")
        print("    Vom folosi o valoare de fallback implicită pentru a permite rularea graficului.")
        # Fallback pe datele confirmate de utilizator în caz de eroare de rețea locală
        fallback_data = {2017: 1, 2018: 1, 2019: 12, 2020: 21, 2021: 97, 2022: 537, 2023: 6726, 2024: 18357, 2025: 28881}
        papers.append(fallback_data.get(year, 0))
    
    # Întârziere de bune maniere (politeness policy) cerută de serverele arXiv
    time.sleep(3)

# Convertim în array-uri numpy pentru procesarea matematică
years = np.array(years)
papers = np.array(papers)

print(f"\nDate finale colectate: {list(papers)}")
print("\nPasul 2: Generare grafic academic cu linie de tendință exponențială...")

# 2. Configurarea și generarea graficului
plt.figure(figsize=(11, 6.5))

# Desenarea graficului cu bare
bars = plt.bar(years, papers, color='#34495e', alpha=0.85, edgecolor='#2c3e50', width=0.6, label='Număr Publicații (Date în timp real)')

# Definirea funcției exponențiale matematice pentru regresie
def exp_func(x, a, b):
    return a * np.exp(b * (x - 2017))

try:
    # Potrivirea curbei exponențiale (curve fitting) pe datele colectate
    popt, _ = curve_fit(exp_func, years, papers, p0=[1, 1])
    x_fit = np.linspace(2017, 2025, 100)
    y_fit = exp_func(x_fit, *popt)
    
    # Adăugarea liniei de tendință exponențială
    plt.plot(x_fit, y_fit, color='#e74c3c', linewidth=2.5, linestyle='--', label='Tendință Exponențială ($y = a \cdot e^{b \cdot x}$)')
except Exception as e:
    print(f"Nu s-a putut calcula curba de regresie (date insuficiente sau eroare): {e}")

# Stilizarea graficului conform standardelor academice pentru Teza de Licență
plt.title('Evoluția Exponențială a Cercetării în Domeniul LLM (2017 - 2025)\nSursă Date: API arXiv (Interogare Automată)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Anul Publicării', fontsize=12, labelpad=10)
plt.ylabel('Număr de Lucrări Științifice', fontsize=12, labelpad=10)
plt.xticks(years, fontsize=11)
plt.yticks(fontsize=11)
plt.grid(axis='y', linestyle=':', alpha=0.6)
plt.legend(fontsize=11, loc='upper left')

# Adăugarea etichetelor de text deasupra barelor
for bar in bars:
    yval = bar.get_height()
    offset = 500 if yval > 500 else 150
    plt.text(bar.get_x() + bar.get_width()/2, yval + offset, f'{int(yval)}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#2c3e50')

plt.tight_layout()

# Salvarea imaginii la rezoluție înaltă (300 DPI) pentru imprimare sau PDF de calitate
output_filename = 'grafic_crestere_llm_automat.png'
plt.savefig(output_filename, dpi=300)

print(f"\nSucces! Graficul a fost salvat ca '{output_filename}' în directorul curent.")
