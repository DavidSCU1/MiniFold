def generate_html_view(pdb_file, output_html="structure.html"):
    """
    Generates an HTML file with embedded py3Dmol viewer.
    """
    try:
        with open(pdb_file, "r") as f:
            pdb_content = f.read()
            
        # Escape backticks or other chars if necessary, but PDB usually is fine.
        # We'll use a template.
        
        pdb_js = pdb_content.replace("`", "\\`")
        html_template = """<!DOCTYPE html>
<html>
<head>
    <title>MiniFold Structure Prediction</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.3/3Dmol-min.js"></script>
    <style>
        body { margin: 0; padding: 0; font-family: sans-serif; }
        #container { width: 100%; height: 100vh; position: relative; }
        #caption { position: absolute; top: 10px; left: 10px; z-index: 10; background: rgba(255,255,255,0.8); padding: 10px; border-radius: 5px; }
    </style>
</head>
<body>
    <div id="caption">
        <h2>MiniFold Result</h2>
        <p>Model: Full Atom Model (Backbone + Sidechains)</p>
        <div style="font-size: 0.9em; margin-top: 5px;">
            <label><input type="radio" name="style" value="cartoon" checked onchange="setStyle(this.value)"> Cartoon + Sticks</label><br>
            <label><input type="radio" name="style" value="sphere" onchange="setStyle(this.value)"> Sphere</label><br>
            <label><input type="radio" name="style" value="surface" onchange="setStyle(this.value)"> Surface</label>
        </div>
    </div>
    <div id="container" class="mol-container"></div>
    
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            var element = document.getElementById('container');
            var config = { backgroundColor: 'white' };
            window.viewer = $3Dmol.createViewer(element, config);
            var pdbData = `__PDB__`;
            
            try {
                window.viewer.addModel(pdbData, 'pdb');
                setStyle('cartoon');
                window.viewer.zoomTo();
                window.viewer.render();
                window.viewer.zoom(1.2, 1000);
            } catch (e) {
                console.error(e);
                element.innerHTML = '<p style="padding:12px;color:#ef4444;font-size:12px;">Failed to render PDB: ' + e + '<br/>Showing raw text below.</p>';
                var pre = document.createElement('pre');
                pre.style.fontFamily = 'monospace';
                pre.style.fontSize = '11px';
                pre.style.whiteSpace = 'pre-wrap';
                pre.style.wordBreak = 'break-word';
                pre.style.background = '#fff';
                pre.style.color = '#111827';
                pre.style.padding = '10px';
                pre.textContent = pdbData;
                element.appendChild(pre);
            }
        });

        function setStyle(style) {
            if (!window.viewer) return;
            window.viewer.removeAllLabels();
            window.viewer.setStyle({}, {}); // Clear
            
            if (style === 'cartoon') {
                // Cartoon for backbone + Sticks for sidechains
                window.viewer.addStyle({}, {cartoon: {color: 'spectrum', thickness: 0.4}});
                // Show sidechains as sticks (excluding backbone atoms usually, but py3Dmol handles overlap well)
                window.viewer.addStyle({}, {stick: {radius: 0.15, colorscheme: 'Jmol'}});
            } else if (style === 'sphere') {
                window.viewer.addStyle({}, {sphere: {scale: 0.3, colorscheme: 'Jmol'}});
            } else if (style === 'surface') {
                window.viewer.addStyle({}, {cartoon: {color: 'spectrum'}});
                window.viewer.addSurface($3Dmol.SurfaceType.VDW, {opacity: 0.7, color: 'white'}, {});
            }
            window.viewer.render();
        }
    </script>
</body>
</html>"""

        html_final = html_template.replace("__PDB__", pdb_js)
        with open(output_html, "w", encoding='utf-8') as f:
            f.write(html_final)
            
        print(f"Visualization saved to {output_html}")
        return True

    except Exception as e:
        print(f"Error generating visualization: {e}")
        return False
