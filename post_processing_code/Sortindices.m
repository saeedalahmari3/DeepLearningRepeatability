function [lx,ly,width,hight] = Sortindices(C)
x = C(:,1);
y = C(:,2);
smallestX = x(1);
smallestY = y(1);
largestX = x(4);
largestY = y(4);
for i =1 : numel(x)
  if smallestX > x(i)
     smallestX = x(i);
  end
  if smallestY > y(i)
      smallestY = y(i);
  end
  if largestX < x(i)
      largestX = x(i);
  end
  if largestY < y(i)
      largestY = y(i);
  end
end
lx = smallestX;
ly = smallestY;
width = largestX - lx;
hight = largestY - ly;

end
