"""
Lunar Mass Driver Orbital Simulation — entry point.

Run:
    python app.py
Then open http://localhost:8050

app.py is intentionally kept import-free so that ProcessPoolExecutor worker
processes, which re-execute the __main__ script on Windows (spawn method),
do not import Dash/Plotly and do not trigger MemoryErrors.
"""

if __name__ == "__main__":
    import sys
    from webapp import app
    debug = "--debug" in sys.argv
    app.run(debug=debug, host="0.0.0.0", port=8050)
