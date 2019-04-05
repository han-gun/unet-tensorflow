load('imdb_denoising.mat');

train_id = find(set == 0);
val_id = find(set == 1);
test_id = find(set == 2);

labels = label;
inputs = input;

%%
sgm = 30;

for i = 1:length(train_id)
    label = labels(:,:,:,train_id(i));
    input = inputs(:,:,:,train_id(i)) + sgm*randn(size(label));
    
    save(num2str(i-1, 'train/label_%04d.mat'), 'label');
    save(num2str(i-1, 'train/input_%04d.mat'), 'input');
    
end

%%
for i = 1:length(val_id)
    label = labels(:,:,:,val_id(i));
    input = inputs(:,:,:,val_id(i)) + sgm*randn(size(label));
    
    save(num2str(i-1, 'val/label_%04d.mat'), 'label');
    save(num2str(i-1, 'val/input_%04d.mat'), 'input'); 
end

%%
for i = 1:length(test_id)
    label = labels(:,:,:,test_id(i));
    input = inputs(:,:,:,test_id(i)) + sgm*randn(size(label));
    
    save(num2str(i-1, 'test/label_%04d.mat'), 'label');
    save(num2str(i-1, 'test/input_%04d.mat'), 'input'); 
end