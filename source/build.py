import os
import yaml
import sys
import shutil
import traceback
import re
from html import unescape
from jinja2 import Environment, FileSystemLoader

# Chart generation imports
import matplotlib.pyplot as plt
import numpy as np
from math import pi
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import requests
from io import BytesIO

def load_data(filename):
    """Load data from YAML file"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, 'data', filename)
    
    with open(data_path, 'r') as f:
        return yaml.safe_load(f)

# --- Chart Generation Functions ---

def load_icon_from_url(url, size=40):
    """Load icon from URL and resize it"""
    if not url:
        return np.ones((size, size, 3), dtype=np.uint8) * 200
        
    try:
        response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        content = response.content
        
        if url.endswith('.svg') or b'<svg' in content[:100]:
            try:
                import cairosvg
                png_data = cairosvg.svg2png(bytestring=content, output_width=size*2, output_height=size*2)
                img = Image.open(BytesIO(png_data))
            except ImportError:
                try:
                    from svglib.svglib import svg2rlg
                    from reportlab.graphics import renderPM
                    drawing = svg2rlg(BytesIO(content))
                    png_data = renderPM.drawToString(drawing, fmt='PNG')
                    img = Image.open(BytesIO(png_data))
                except ImportError:
                    print(f"Note: SVG support requires 'cairosvg' or 'svglib'.")
                    return np.ones((size, size, 3), dtype=np.uint8) * 200
        else:
            img = Image.open(BytesIO(content))
        
        if img.mode == 'RGBA':
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'RGBA':
                background.paste(img, mask=img.split()[3] if len(img.split()) == 4 else None)
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
            
        img.thumbnail((size, size), Image.Resampling.LANCZOS)
        return np.array(img)
    except Exception as e:
        print(f"Warning: Could not load icon from {url}: {e}")
        return np.ones((size, size, 3), dtype=np.uint8) * 200

def generate_radar_chart(data_items, title, filename, output_dir):
    """Generate a single radar chart with icons"""
    categories = [item['name'] for item in data_items]
    values = [item['value'] for item in data_items]
    icons = {item['name']: item.get('icon') for item in data_items}
    
    n_categories = len(categories)
    angles = [n / float(n_categories) * 2 * pi for n in range(n_categories)]
    angles += angles[:1]
    values += values[:1]
    
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
    
    ax.plot(angles, values, '-', linewidth=2, color='steelblue')
    ax.fill(angles, values, alpha=0.3, color='steelblue')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])
    ax.set_ylim(0, 110)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=9)
    ax.grid(True, linestyle='--', linewidth=0.5)
    
    plt.title(title, size=16, fontweight='bold', pad=30)
    
    icon_values = values[:-1]
    
    for i, (category, angle, value) in enumerate(zip(categories, angles[:-1], icon_values)):
        label_radius = value + 15
        ax.text(angle, label_radius, category, 
               ha='center', va='center',
               fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
    
    plt.tight_layout()
    fig.canvas.draw()
    
    for i, (category, angle, value) in enumerate(zip(categories, angles[:-1], icon_values)):
        icon_radius = 110
        if icons[category]:
            try:
                icon_img = load_icon_from_url(icons[category], size=60)
                icon_img_pil = Image.fromarray(icon_img)
                imagebox = OffsetImage(icon_img_pil, zoom=0.5)
                ab = AnnotationBbox(imagebox, (angle, icon_radius),
                                    xycoords=ax.transData,
                                    frameon=False, pad=0,
                                    box_alignment=(0.5, 0.5),
                                    zorder=10)
                ax.add_artist(ab)
            except Exception as e:
                print(f"Warning: Could not add icon for {category}: {e}")
    
    svg_path = os.path.join(output_dir, f'{filename}.svg')
    plt.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')
    plt.close()
    return svg_path

# --- Rendering ---

def render_template(template_name, output_path, **kwargs):
    """Generic function to render a template"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    template_dir = os.path.join(script_dir, 'templates')
    
    env = Environment(loader=FileSystemLoader(template_dir))
    
    # Add markdown filter
    try:
        import markdown

        def markdown_to_html(text):
            return markdown.markdown(text)

        def markdown_inline(text):
            html = markdown_to_html(text).strip()
            if html.startswith('<p>') and html.endswith('</p>'):
                html = html[3:-4]
            return html

        def markdown_text(text):
            html = markdown_inline(text)
            text_only = re.sub(r'<[^>]+>', '', html)
            return unescape(text_only)

        env.filters['markdown'] = markdown_to_html
        env.filters['markdown_inline'] = markdown_inline
        env.filters['markdown_text'] = markdown_text
    except ImportError:
        print("Warning: 'markdown' module not found. Markdown rendering will be disabled.")
        env.filters['markdown'] = lambda text: text
        env.filters['markdown_inline'] = lambda text: text
        env.filters['markdown_text'] = lambda text: text

    template = env.get_template(template_name)
    
    rendered_content = template.render(**kwargs)
    
    with open(output_path, 'w') as f:
        f.write(rendered_content)
    
    print(f"✓ Generated {output_path}")

def main():
    print("Starting bio update process...")
    
    # Determine paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    docs_dir = os.path.join(project_root, 'docs')
    assets_dir = os.path.join(docs_dir, 'assets')
    
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir)

    # Load Data
    try:
        profile = load_data('profile.yaml')
        projects = load_data('projects.yaml')
        skills = load_data('skills.yaml')
        experience = load_data('experience.yaml')

        # Process experience data to generate Mermaid syntax
        from datetime import datetime
        today = datetime.now().strftime("%d-%m-%Y")
        
        for section in experience.get('sections', []):
            for item in section.get('items', []):
                # Format: [criteria], start, [end|duration]
                # If criteria exists, prepend it with a comma
                criteria = f"{item['criteria']}, " if 'criteria' in item else ""
                start = item['start']
                
                if item.get('current'):
                    end = today
                    item['data'] = f"{criteria}{start}, {end}"
                elif item.get('end'):
                    item['data'] = f"{criteria}{start}, {item['end']}"
                elif item.get('duration'):
                    duration = item['duration']
                    item['data'] = f"{criteria}{start}, {duration}"
                else:
                    # Fallback if neither current nor duration is set (though schema implies one should be)
                    item['data'] = f"{criteria}{start}, {today}"
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    # Step 1: Generate Charts directly to docs/assets
    print("\n--- Generating Charts ---")
    print(f"✓ Experience Gantt Chart data prepared (rendered client-side)")
    try:
        for key, info in skills.items():
            svg_path = generate_radar_chart(
                info['items'],
                info['title'],
                info['filename'],
                output_dir=assets_dir
            )
            print(f"✓ {info['title']} chart generated at {svg_path}")
    except Exception as e:
        traceback.print_exc()
        print(f"Error generating charts: {e}")

    # Step 2: Render README
    print("\n--- Generating README.md ---")
    try:
        readme_path = os.path.join(project_root, 'README.md')
        render_template('README.md.j2', readme_path, profile=profile, projects=projects, skills=skills, experience=experience)
    except Exception as e:
        traceback.print_exc()
        print(f"Error rendering README: {e}")
        sys.exit(1)

    # Step 3: Render Website (docs/index.html)
    print("\n--- Generating Website (docs/index.html) ---")
    try:
        index_path = os.path.join(docs_dir, 'index.html')
        render_template('index.html.j2', index_path, profile=profile, projects=projects, skills=skills, experience=experience)
    except Exception as e:
        traceback.print_exc()
        print(f"Error rendering Website: {e}")
        sys.exit(1)
        
    print("\nSUCCESS: Bio updated successfully!")

if __name__ == "__main__":
    main()
