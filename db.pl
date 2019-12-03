loves(romeo,juliet).

loves(juliet,romeo) :- loves(romeo,juliet).

male(albert).
male(bob).
male(bill).
male(carl).
male(charlie).
male(dan).
male(edward).

female(alice).
female(betsy).
female(diana).


happy(albert).
happy(alice).
happy(bob).
happy(bill).
with_albert(alice).

runs(albert) :-
 happy(albert).

dances(alice) :-
    happy(alice),
    with_albert(alice).

does_alice_dance :- dances(alice),
    write('WQhen alice is happu and with albert she dances').


swims(bob) :- 
    happy(bob),
    near_water(bob).

