%Implementation of DAG approach. 
function predictedValue = DAG(models,list,testVar,tableWithGeneralizationCap)    
    l = transpose(list);
    numcats = length(l);
    if numcats==2 %stopping criteria
        class1 = list(1);
        class2 = list(2);
        %select index of corresponding 1v1 predictor
        indexOfPredictor = tableWithGeneralizationCap.Index(tableWithGeneralizationCap{:,1:1}== class1 & tableWithGeneralizationCap{:,2:2} == class2); 
        %predict value for the instance
        predictedValue = predict(models{indexOfPredictor},testVar);
    else         
        [~, min_idx] = min(tableWithGeneralizationCap{:,3});
        class1 = tableWithGeneralizationCap{min_idx,1:1};
        class2 = tableWithGeneralizationCap{min_idx,2:2};
        indexOfPredictor = tableWithGeneralizationCap{min_idx,4:4};
        value = predict(models{indexOfPredictor},testVar);
        if value==class1
            %remove every classifier that uses class 1
            listNew = list(list~= class2);
            toDelete = tableWithGeneralizationCap.Class1 == class2 | tableWithGeneralizationCap.Class2 == class2 ;
            tableWithGeneralizationCap(toDelete,:) = [];
            predictedValue = DAG(models,listNew,testVar,tableWithGeneralizationCap);
        else
            %remove every classifier that uses class 2
            listNew = list(list~= class1);
            toDelete = tableWithGeneralizationCap.Class1 == class1 | tableWithGeneralizationCap.Class2 == class1 ;
            tableWithGeneralizationCap(toDelete,:) = [];
            predictedValue = DAG(models,listNew,testVar,tableWithGeneralizationCap);
        end
    end             
end
