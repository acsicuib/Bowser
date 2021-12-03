# Deep MARIO vs Bowser 
## A project.

### On
- Python 3.9.7 (default, Nov  4 2021, 13:09:39)
  
  
### Provisional steps

```bash 
source marioenv/bin/activate


#on MARIO/multi-agent-policies/ folder)
export PYTHONPATH="/home/isaac/projects/Bowser:/home/isaac/projects/Bowser/environment:$PYTHONPATH"
python3 main.py

scp isaac@cloudlab:/home/isaac/projects/Bowser/results/results_SmallFog_X/samples.csv .
```