# somalab

The thought process for solving is in thoughts.org (also at the bottom of this incase emacs is problematic) 

running.py is the main code with the curl command and the results mentioned at the bottom. the test for this result is included in test.json 
medical_dialogue_dataset.json is the file that was used for the fine tuning with the help of datastruct.py. 
tuning.py is the main for using Llama and  lotok.py is for the tokenization. 

\\ trial.c was some initial attempt at trying something different with C, mostly redundant stuff unless you also know how to approach it from there so just added it incase. 

\\  website problem: On phones, atleast Iphones, i noticed that text was being cutoff when reading through the somalab website. I have attached a pic in the repo itself and imo its important info being cutoff and so just wanted to bring it to your notice. from what i know about webflow, i think it might be the breakpoints for mobile devices that would need to be toggled with it fix it I assume. Hope that helps. 

---------------

#from thoughts.org 


With the main goals as mentioned in the document being
1) accuracy
2) patient specific capabailities
3) latency

   hence goal is to create the most accurate negativity analysis system.

So I would assign ratings  on the following scales:
Overall negativity (0-10): 
Perceived judgment/criticism (0-5):
Potential for anxiety/stress (0-5): 
Empathy/rapport building (-5 to +5):

And then wanted to flow in the form of txt files --> running.py + test.json --> input results --> fine tuning using llama ----> final results


Effectively then out of the three ideas for improvement mentioned in the document
i could with this system implement a sentiment analysis API, fine tuning, and incorporate case specific guidelines based on patient context.
the main thing that could be furthered improved from this then remains implementing a mixture of experts or a tree of thought approach, which would be a more advanced, multi-path analysis. 
