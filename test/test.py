import sys, pathlib

# Ensure backend package is importable when running from repo root
repo_root = pathlib.Path(__file__).resolve().parents[1]
backend_path = repo_root / "backend"
if str(backend_path) not in sys.path:
	sys.path.insert(0, str(backend_path))

from rag_app.rag_embeddings import get_embedding

def main():
	text = "quantum transformer models for tomography"
	print(f"Generating embedding for: {text}")
	try:
		vec = get_embedding(text)
	except Exception as e:
		print("ERROR generating embedding:", e)
		return
	print("Vector length:", len(vec))
	print("First 8 dims:", [round(x, 5) for x in vec[:8]])
	# Basic sanity: non-zero variance (unless using fake embed)
	uniq = {round(x, 6) for x in vec}
	print("Unique values count:", len(uniq))
	print("Full vector:", vec)   # add after getting vec

if __name__ == "__main__":
	main()