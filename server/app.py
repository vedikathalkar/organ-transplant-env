import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))
from main import app

def main():
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=7860)

if __name__ == '__main__':
    main()