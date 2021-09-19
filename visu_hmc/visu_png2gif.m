clear
istart = 1;
nsave  = 90;
dtim   = 0.08;
st     = 0.6;
iskip  = 1;
inputname = 'PT_HMC_Atg_1023x1023';
filename = 'PT_HMC_Atg_1023x1023.gif';
for isave = istart:iskip:nsave
    fname = [inputname '_' int2str(isave) '.png'];
    im2 = imread(fname);
    im  = imresize(im2, st);
    [imind,cm] = rgb2ind(im,256);
    % Write to the GIF File
    if isave == istart
        imwrite(imind,cm,filename,'gif', 'Loopcount',inf,'DelayTime',dtim);
    else
        imwrite(imind,cm,filename,'gif','WriteMode','append','DelayTime',dtim);
    end
end
