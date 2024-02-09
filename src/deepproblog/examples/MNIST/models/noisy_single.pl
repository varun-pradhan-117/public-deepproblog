nn(mnist_net,[X],Y,[0,1,2,3,4,5,6,7,8,9]) :: digit(X,Y).

t(0.1) :: noisy.

1/10::uniform(X,0); 1/10::uniform(X,1); 1/10::uniform(X,2); 1/10::uniform(X,3); 1/10::uniform(X,4); 1/10::uniform(X,5); 1/10::uniform(X,6); 1/10::uniform(X,7); 1/10::uniform(X,8); 1/10::uniform(X,9).

detection_noisy(X,Y) :- noisy, uniform(X,Y).
detection_noisy(X,Y) :- \+noisy, digit(X,Y).
