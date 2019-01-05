import Cocoa
import CreateML
import Foundation

let Training = URL(fileURLWithPath: "/Users/aritropaul/Documents/Codes/iOS/Playgrounds/MLPlaygrounds/Titanic/train.csv")
let Test =  URL(fileURLWithPath: "/Users/aritropaul/Documents/Codes/iOS/Playgrounds/MLPlaygrounds/Titanic/test.csv")

let passengerDataTrain = try MLDataTable(contentsOf: Training).dropMissing()
let passengerDataTest = try MLDataTable(contentsOf: Test)
let survival = try MLLinearRegressor(trainingData: passengerDataTrain, targetColumn: "Survived", parameters:  MLLinearRegressor.ModelParameters(maxIterations: 100))
let metrics = try survival.predictions(from: passengerDataTest)
let submissionData = try MLDataTable(namedColumns: ["PassengerId": passengerDataTest["PassengerId"],"Survived":metrics > 0.5])
try survival.write(toFile: "/Users/aritropaul/Documents/Codes/iOS/Playgrounds/MLPlaygrounds/Titanic/TitanicSurvival")
var csvText = "PassengerId,Survived\n"
for rows in submissionData.rows{
    print(rows)
    let newLine = "\(rows.values[0].intValue!),\(rows.values[1].intValue!)\n"
    csvText.append(contentsOf: newLine)
}

do {
    try csvText.write(to: URL(fileURLWithPath: "/Users/aritropaul/Documents/Codes/iOS/Playgrounds/MLPlaygrounds/Titanic/submission.csv"), atomically: true, encoding: .utf8)
} catch {
    print("\(error.localizedDescription)")
}


