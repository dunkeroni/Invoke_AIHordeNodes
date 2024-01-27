# Invoke_AIHordeNodes
Nodes to interface Invoke workflows with the AI Horde  
![image](https://github.com/dunkeroni/Invoke_AIHordeNodes/assets/3298737/e5d14f1b-ac0c-4675-8a91-cf430488b598)

## Usage:  
git clone this repository into your invokeAI/nodes/ folder.  
Create a copy of the `userconfig.TEMPLATE.cfg` file and name it `userconfig.cfg` and provide your API Key in it. If you do not do this step, then it will be done for you when you launch Invoke and the apikey will be set to anonymous.  
Changes to the configuration require Invoke to be restarted.  

The available models list is loaded when Invoke is launched. Loading past workflows may include an input list that conflicts with this one if the horde adds new models. If that happens, you will get an error during invocation. To fix this, delete the horde request node and add it back into the workflow.  
At some point this interaction may be changed with a model option that accepts a collection of strings. There will be a node that can fuzzy search for model names in the active model list in addition to a node that offers a dropdown list that will forever and always have this bug.  

## Workflows:  
The Horde request nodes are designed to allow optional settings extensions. If settings are not provided, then the server will use its own defaults. For that reason, it is best to always include General and Advanced settings, however they can be omitted for simple requests.
![image](https://github.com/dunkeroni/Invoke_AIHordeNodes/assets/3298737/b21cf1e7-c76f-49b9-8444-298a3c640ca3)  


The fastest way to find the settings node is to click and drag one of the inputs to an empty location in the editor. This will bring up a list of all the nodes that can be attached there.  
![image](https://github.com/dunkeroni/Invoke_AIHordeNodes/assets/3298737/16905aa6-a127-4cb6-8d56-5fbff01bf0f3)
If you use General Settings input, remember to attach a random integer node to the seed, otherwise all of your requests will use `seed="0"`  
If you do not use the General Settings input, then a random integer will be chosen.  
