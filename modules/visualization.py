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
        <p>Model: Backbone Predictor</p>
        <p>Style: Trace (Spectrum Color)</p>
    </div>
    <div id="container" class="mol-container"></div>
    
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            var element = document.getElementById('container');
            var config = { backgroundColor: 'white' };
            var viewer = $3Dmol.createViewer(element, config);
            var pdbData = `__PDB__`;
            try {
                viewer.addModel(pdbData, 'pdb');
                // Use trace style which is robust for incomplete backbones (missing O atoms)
                viewer.setStyle({}, {stick: {radius: 0.15}, sphere: {scale: 0.3}}); 
                // Also add a cartoon-like trace
                viewer.addStyle({}, {cartoon: {color: 'spectrum', thickness: 0.5, opacity: 0.8}});
                viewer.zoomTo();
                viewer.render();
                viewer.zoom(1.2, 1000);
            } catch (e) {
                console.error(e);
                element.innerHTML = '<p style="padding:20px;color:red;">Failed to render PDB: ' + e + '</p>';
            }
        });
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
