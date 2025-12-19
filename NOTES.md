# General notes and ideas that did not make it to the roadmap specs yet.

Why are the seed strategies fixed? They could be in the program_db. Then you do not need to start over with the same seed strategy every time. You could just load it from the db (by adding a hash to make sure you do not add the same strategy twice). We could have a script that populates the program_db with initial seed strategies. And it should be easy to add new seed strategies later. The CLI could have the start strategy as on optional argument.

People could contribute seed strategies to the program_db. People could also contribute strategies that were evolved by the system (and found to be good).

How to deal with missing librairies in the strategy code? (e.g. talib) Maybe have a list of allowed libraries and check the code against that list before running it. Maybe if a strategy needs a libary it should be queued for human review before being compiled and run, so that the human can install the library if needed.

Run the strategy in a sandboxed environment to avoid security issues. Maybe this would allow strategies to install their own libraries safely.

Store the model used to generate the strategies alongside the strategies themselves, so that you can reproduce or further evolve them later.

Have a UI/website where you can see the system in action and browse the strategies. An ancestor tree viewer would be nice. postgres/supabase could be used to store the strategies and their metadata. Supabase would be nice to auto update the ui when new strategies are added.

Only the chosen strategy is running the test phase. Need to check if this is correct. Why does not every strategy have a test run value? Why is the strategy not rejected if the test run deviates too much from the run value?

Have a good README with references to papers. Add the MIT license file.

StratGenesis – Meaning: Conveys “strategy genesis,” i.e. the birth of new trading strategies. It suggests a system that continuously generates and evolves strategies from scratch. Plan for renaming the project.

Test what we have now before going further with phase 16 adding Research and Data Agents.

If people would contribute how do we deal with the data that a strategy uses? Probably need to keep it local and if you want to run a strategy you have to get the data yourself (the data collecting agents could help with that). basically you could not use strategies that use data you do not have.


