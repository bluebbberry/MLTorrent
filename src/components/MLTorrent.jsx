import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw, Users, Database, Zap, Network, Download, Upload, Brain, Shield, TrendingUp, BarChart3, Settings, Target, Clock, Award } from 'lucide-react';

const MLTorrent = () => {
    const [isTraining, setIsTraining] = useState(false);
    const [currentEpoch, setCurrentEpoch] = useState(0);
    const [maxEpochs, setMaxEpochs] = useState(50);
    const [globalAccuracy, setGlobalAccuracy] = useState(0);
    const [peers, setPeers] = useState([]);
    const [networkActivity, setNetworkActivity] = useState([]);
    const [totalDataShared, setTotalDataShared] = useState(0);
    const [modelFragments, setModelFragments] = useState(0);
    const [trainingData, setTrainingData] = useState([]);
    const [testData, setTestData] = useState([]);
    const [globalModel, setGlobalModel] = useState(null);
    const [trainingComplete, setTrainingComplete] = useState(false);
    const [trainingHistory, setTrainingHistory] = useState([]);
    const [peerPerformanceHistory, setPeerPerformanceHistory] = useState([]);
    const [startTime, setStartTime] = useState(null);
    const [endTime, setEndTime] = useState(null);
    const [showPerformanceOverview, setShowPerformanceOverview] = useState(false);
    const intervalRef = useRef(null);

    // Generate synthetic training data (2D classification problem)
    const generateTrainingData = () => {
        const data = [];
        const numSamples = 1000;

        for (let i = 0; i < numSamples; i++) {
            const x1 = Math.random() * 10 - 5;
            const x2 = Math.random() * 10 - 5;
            // Create a simple decision boundary: y = 1 if x1 + x2 > 0, else 0
            const y = x1 + x2 + Math.random() * 0.5 - 0.25 > 0 ? 1 : 0;
            data.push({ x1, x2, y });
        }

        return data;
    };

    // Generate test data (separate from training)
    const generateTestData = () => {
        const data = [];
        const numSamples = 200;

        for (let i = 0; i < numSamples; i++) {
            const x1 = Math.random() * 10 - 5;
            const x2 = Math.random() * 10 - 5;
            const y = x1 + x2 + Math.random() * 0.5 - 0.25 > 0 ? 1 : 0;
            data.push({ x1, x2, y });
        }

        return data;
    };

    // Simple linear classifier
    class SimpleLinearClassifier {
        constructor() {
            this.weights = [Math.random() - 0.5, Math.random() - 0.5]; // w1, w2
            this.bias = Math.random() - 0.5;
        }

        predict(x1, x2) {
            const logit = this.weights[0] * x1 + this.weights[1] * x2 + this.bias;
            return 1 / (1 + Math.exp(-logit)); // sigmoid
        }

        train(data, learningRate = 0.01) {
            const gradients = [0, 0, 0]; // for w1, w2, bias

            for (const sample of data) {
                const prediction = this.predict(sample.x1, sample.x2);
                const error = sample.y - prediction;

                gradients[0] += error * sample.x1;
                gradients[1] += error * sample.x2;
                gradients[2] += error;
            }

            // Update weights
            this.weights[0] += learningRate * gradients[0] / data.length;
            this.weights[1] += learningRate * gradients[1] / data.length;
            this.bias += learningRate * gradients[2] / data.length;

            return gradients;
        }

        evaluate(data) {
            let correct = 0;
            let totalLoss = 0;

            for (const sample of data) {
                const prediction = this.predict(sample.x1, sample.x2);
                const binaryPrediction = prediction > 0.5 ? 1 : 0;
                if (binaryPrediction === sample.y) correct++;

                // Calculate cross-entropy loss
                const loss = -(sample.y * Math.log(prediction + 1e-15) + (1 - sample.y) * Math.log(1 - prediction + 1e-15));
                totalLoss += loss;
            }

            return {
                accuracy: correct / data.length,
                loss: totalLoss / data.length
            };
        }

        getWeights() {
            return {
                weights: [...this.weights],
                bias: this.bias
            };
        }

        setWeights(weights) {
            this.weights = [...weights.weights];
            this.bias = weights.bias;
        }
    }

    // Initialize peers with their own data shards and models
    useEffect(() => {
        const fullData = generateTrainingData();
        const testDataSet = generateTestData();
        setTrainingData(fullData);
        setTestData(testDataSet);

        // Create 5 peers with different data shards
        const peerCount = 5;
        const shardSize = Math.floor(fullData.length / peerCount);

        const initialPeers = [];
        for (let i = 0; i < peerCount; i++) {
            const start = i * shardSize;
            const end = i === peerCount - 1 ? fullData.length : start + shardSize;
            const peerData = fullData.slice(start, end);

            const peer = {
                id: i + 1,
                name: ['SmartHome-Berlin', 'IoT-Munich', 'EdgeDevice-Hamburg', 'SmartCity-Frankfurt', 'Home-Stuttgart'][i],
                model: new SimpleLinearClassifier(),
                data: peerData,
                accuracy: 0,
                loss: 0,
                status: 'active',
                contribution: 0.8 + Math.random() * 0.2,
                lastUpdate: Date.now(),
                downloadSpeed: 0,
                uploadSpeed: 0,
                totalUpdates: 0
            };

            // Initial accuracy calculation
            const evaluation = peer.model.evaluate(peer.data);
            peer.accuracy = evaluation.accuracy;
            peer.loss = evaluation.loss;
            initialPeers.push(peer);
        }

        setPeers(initialPeers);

        // Initialize global model
        const globalMod = new SimpleLinearClassifier();
        setGlobalModel(globalMod);
        const globalEval = globalMod.evaluate(testDataSet);
        setGlobalAccuracy(globalEval.accuracy);

        // Initialize history
        setTrainingHistory([{
            epoch: 0,
            globalAccuracy: globalEval.accuracy,
            globalLoss: globalEval.loss,
            avgPeerAccuracy: initialPeers.reduce((sum, p) => sum + p.accuracy, 0) / initialPeers.length
        }]);

        setPeerPerformanceHistory([initialPeers.map(p => ({
            peerId: p.id,
            accuracy: p.accuracy,
            loss: p.loss,
            epoch: 0
        }))]);
    }, []);

    // Federated averaging function
    const federatedAveraging = (peerModels, contributions) => {
        const totalContribution = contributions.reduce((sum, c) => sum + c, 0);
        const avgWeights = [0, 0];
        let avgBias = 0;

        peerModels.forEach((model, i) => {
            const weight = contributions[i] / totalContribution;
            const modelWeights = model.getWeights();
            avgWeights[0] += modelWeights.weights[0] * weight;
            avgWeights[1] += modelWeights.weights[1] * weight;
            avgBias += modelWeights.bias * weight;
        });

        return {
            weights: avgWeights,
            bias: avgBias
        };
    };

    const performTrainingRound = () => {
        setPeers(prevPeers => {
            const updatedPeers = [...prevPeers];

            // Each peer trains on its local data
            updatedPeers.forEach(peer => {
                if (peer.status === 'active') {
                    // Local training
                    peer.model.train(peer.data);
                    const evaluation = peer.model.evaluate(peer.data);
                    peer.accuracy = evaluation.accuracy;
                    peer.loss = evaluation.loss;
                    peer.totalUpdates++;

                    // Simulate network activity
                    peer.uploadSpeed = Math.random() * 100 + 50;
                    peer.downloadSpeed = Math.random() * 100 + 50;
                    peer.lastUpdate = Date.now();
                }
            });

            // Federated averaging (global model update)
            const activeModels = updatedPeers.filter(p => p.status === 'active').map(p => p.model);
            const contributions = updatedPeers.filter(p => p.status === 'active').map(p => p.contribution);

            if (activeModels.length > 0) {
                const avgWeights = federatedAveraging(activeModels, contributions);
                globalModel.setWeights(avgWeights);

                // Update global accuracy on test data
                const globalEval = globalModel.evaluate(testData);
                setGlobalAccuracy(globalEval.accuracy);

                // Update training history
                setTrainingHistory(prev => [...prev, {
                    epoch: currentEpoch + 1,
                    globalAccuracy: globalEval.accuracy,
                    globalLoss: globalEval.loss,
                    avgPeerAccuracy: updatedPeers.reduce((sum, p) => sum + p.accuracy, 0) / updatedPeers.length
                }]);

                // Update peer performance history
                setPeerPerformanceHistory(prev => [...prev, updatedPeers.map(p => ({
                    peerId: p.id,
                    accuracy: p.accuracy,
                    loss: p.loss,
                    epoch: currentEpoch + 1
                }))]);

                // Distribute updated model back to peers (gossip protocol simulation)
                updatedPeers.forEach(peer => {
                    if (peer.status === 'active' && Math.random() > 0.3) {
                        // Peer receives partial global model update
                        const currentWeights = peer.model.getWeights();
                        const globalWeights = globalModel.getWeights();

                        // Blend local and global models
                        const blendFactor = 0.3;
                        peer.model.setWeights({
                            weights: [
                                currentWeights.weights[0] * (1 - blendFactor) + globalWeights.weights[0] * blendFactor,
                                currentWeights.weights[1] * (1 - blendFactor) + globalWeights.weights[1] * blendFactor
                            ],
                            bias: currentWeights.bias * (1 - blendFactor) + globalWeights.bias * blendFactor
                        });
                    }
                });
            }

            // Random peer status changes
            updatedPeers.forEach(peer => {
                if (Math.random() < 0.05) {
                    peer.status = peer.status === 'active' ? 'syncing' : 'active';
                }
            });

            return updatedPeers;
        });

        // Add network activity
        setNetworkActivity(prev => {
            const newActivities = [];
            const activePeers = peers.filter(p => p.status === 'active');

            // Generate 2-4 network activities per round
            for (let i = 0; i < Math.floor(Math.random() * 3) + 2; i++) {
                if (activePeers.length >= 2) {
                    const from = activePeers[Math.floor(Math.random() * activePeers.length)];
                    const to = activePeers[Math.floor(Math.random() * activePeers.length)];

                    if (from.id !== to.id) {
                        newActivities.push({
                            id: Date.now() + i,
                            from: from.id,
                            to: to.id,
                            type: Math.random() > 0.4 ? 'model' : 'gradient',
                            size: Math.floor(Math.random() * 150) + 50,
                            timestamp: Date.now()
                        });
                    }
                }
            }

            return [...prev.slice(-15), ...newActivities];
        });

        setTotalDataShared(prev => prev + Math.floor(Math.random() * 300) + 200);
        setModelFragments(prev => prev + Math.floor(Math.random() * 8) + 3);
    };

    const startTraining = () => {
        setIsTraining(true);
        setTrainingComplete(false);
        setShowPerformanceOverview(false);
        setStartTime(Date.now());

        intervalRef.current = setInterval(() => {
            setCurrentEpoch(prev => {
                const newEpoch = prev + 1;
                if (newEpoch >= maxEpochs) {
                    // Training complete
                    setIsTraining(false);
                    setTrainingComplete(true);
                    setEndTime(Date.now());
                    setShowPerformanceOverview(true);
                    clearInterval(intervalRef.current);
                    return newEpoch;
                }
                return newEpoch;
            });
            performTrainingRound();
        }, 1500); // Training round every 1.5 seconds
    };

    const stopTraining = () => {
        setIsTraining(false);
        setEndTime(Date.now());
        if (intervalRef.current) {
            clearInterval(intervalRef.current);
        }
    };

    const resetTraining = () => {
        stopTraining();
        setCurrentEpoch(0);
        setGlobalAccuracy(0);
        setNetworkActivity([]);
        setTotalDataShared(0);
        setModelFragments(0);
        setTrainingComplete(false);
        setShowPerformanceOverview(false);
        setStartTime(null);
        setEndTime(null);
        setTrainingHistory([]);
        setPeerPerformanceHistory([]);

        // Reinitialize everything
        const fullData = generateTrainingData();
        const testDataSet = generateTestData();
        setTrainingData(fullData);
        setTestData(testDataSet);

        const peerCount = 5;
        const shardSize = Math.floor(fullData.length / peerCount);

        const resetPeers = [];
        for (let i = 0; i < peerCount; i++) {
            const start = i * shardSize;
            const end = i === peerCount - 1 ? fullData.length : start + shardSize;
            const peerData = fullData.slice(start, end);

            const peer = {
                id: i + 1,
                name: ['SmartHome-Berlin', 'IoT-Munich', 'EdgeDevice-Hamburg', 'SmartCity-Frankfurt', 'Home-Stuttgart'][i],
                model: new SimpleLinearClassifier(),
                data: peerData,
                accuracy: 0,
                loss: 0,
                status: 'active',
                contribution: 0.8 + Math.random() * 0.2,
                lastUpdate: Date.now(),
                downloadSpeed: 0,
                uploadSpeed: 0,
                totalUpdates: 0
            };

            const evaluation = peer.model.evaluate(peer.data);
            peer.accuracy = evaluation.accuracy;
            peer.loss = evaluation.loss;
            resetPeers.push(peer);
        }

        setPeers(resetPeers);

        const globalMod = new SimpleLinearClassifier();
        setGlobalModel(globalMod);
        const globalEval = globalMod.evaluate(testDataSet);
        setGlobalAccuracy(globalEval.accuracy);

        // Initialize history
        setTrainingHistory([{
            epoch: 0,
            globalAccuracy: globalEval.accuracy,
            globalLoss: globalEval.loss,
            avgPeerAccuracy: resetPeers.reduce((sum, p) => sum + p.accuracy, 0) / resetPeers.length
        }]);

        setPeerPerformanceHistory([resetPeers.map(p => ({
            peerId: p.id,
            accuracy: p.accuracy,
            loss: p.loss,
            epoch: 0
        }))]);
    };

    const getStatusColor = (status) => {
        switch(status) {
            case 'active': return 'text-green-400';
            case 'syncing': return 'text-yellow-400';
            case 'downloading': return 'text-blue-400';
            default: return 'text-gray-400';
        }
    };

    const getStatusIcon = (status) => {
        switch(status) {
            case 'active': return <Zap className="w-4 h-4" />;
            case 'syncing': return <RotateCcw className="w-4 h-4 animate-spin" />;
            case 'downloading': return <Download className="w-4 h-4" />;
            default: return <Users className="w-4 h-4" />;
        }
    };

    const formatDuration = (ms) => {
        const seconds = Math.floor(ms / 1000);
        const minutes = Math.floor(seconds / 60);
        const hours = Math.floor(minutes / 60);

        if (hours > 0) {
            return `${hours}h ${minutes % 60}m ${seconds % 60}s`;
        } else if (minutes > 0) {
            return `${minutes}m ${seconds % 60}s`;
        } else {
            return `${seconds}s`;
        }
    };

    const getTrainingStats = () => {
        if (trainingHistory.length === 0) return null;

        const initialAccuracy = trainingHistory[0].globalAccuracy;
        const finalAccuracy = trainingHistory[trainingHistory.length - 1].globalAccuracy;
        const improvement = finalAccuracy - initialAccuracy;
        const bestAccuracy = Math.max(...trainingHistory.map(h => h.globalAccuracy));
        const bestEpoch = trainingHistory.find(h => h.globalAccuracy === bestAccuracy)?.epoch || 0;

        return {
            initialAccuracy,
            finalAccuracy,
            improvement,
            bestAccuracy,
            bestEpoch,
            convergence: improvement > 0.01 ? 'Good' : improvement > 0.005 ? 'Moderate' : 'Limited'
        };
    };

    const stats = getTrainingStats();

    return (
        <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 text-white p-6">
            <div className="max-w-7xl mx-auto">
                {/* Header */}
                <div className="text-center mb-8">
                    <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
                        MLTorrent
                    </h1>
                    <p className="text-xl text-gray-300 mb-6">
                        Distributed Machine Learning via P2P Networks
                    </p>

                    {/* Training Controls */}
                    <div className="flex flex-col sm:flex-row justify-center gap-4 mb-8">
                        <div className="flex items-center gap-2">
                            <Settings className="w-5 h-5 text-gray-400" />
                            <label className="text-sm text-gray-300">Max Epochs:</label>
                            <input
                                type="number"
                                value={maxEpochs}
                                onChange={(e) => setMaxEpochs(Math.max(1, Math.min(200, parseInt(e.target.value) || 1)))}
                                disabled={isTraining}
                                className="w-20 px-2 py-1 bg-white/10 border border-white/20 rounded text-white text-center disabled:opacity-50"
                                min="1"
                                max="200"
                            />
                        </div>
                        <div className="flex gap-2">
                            <button
                                onClick={isTraining ? stopTraining : startTraining}
                                disabled={currentEpoch >= maxEpochs && !isTraining}
                                className={`flex items-center gap-2 px-6 py-3 rounded-lg font-semibold transition-all disabled:opacity-50 ${
                                    isTraining
                                        ? 'bg-red-600 hover:bg-red-700'
                                        : 'bg-green-600 hover:bg-green-700'
                                }`}
                            >
                                {isTraining ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
                                {isTraining ? 'Stop Training' : 'Start Training'}
                            </button>
                            <button
                                onClick={resetTraining}
                                className="flex items-center gap-2 px-6 py-3 rounded-lg font-semibold bg-gray-600 hover:bg-gray-700 transition-all"
                            >
                                <RotateCcw className="w-5 h-5" />
                                Reset
                            </button>
                        </div>
                    </div>

                    {/* Progress Bar */}
                    <div className="max-w-2xl mx-auto mb-6">
                        <div className="flex justify-between text-sm text-gray-300 mb-2">
                            <span>Training Progress</span>
                            <span>{currentEpoch}/{maxEpochs} epochs</span>
                        </div>
                        <div className="w-full bg-gray-700 rounded-full h-2">
                            <div
                                className="bg-gradient-to-r from-green-500 to-blue-500 h-2 rounded-full transition-all duration-300"
                                style={{ width: `${(currentEpoch / maxEpochs) * 100}%` }}
                            />
                        </div>
                    </div>

                    {/* Training Complete Banner */}
                    {trainingComplete && (
                        <div className="bg-gradient-to-r from-green-500/20 to-blue-500/20 border border-green-500/50 rounded-lg p-4 mb-6">
                            <div className="flex items-center justify-center gap-2 mb-2">
                                <Award className="w-6 h-6 text-green-400" />
                                <span className="text-xl font-semibold text-green-400">Training Complete!</span>
                            </div>
                            <p className="text-gray-300">
                                Training completed in {endTime && startTime ? formatDuration(endTime - startTime) : 'Unknown time'}
                            </p>
                        </div>
                    )}
                </div>

                {/* Performance Overview */}
                {showPerformanceOverview && stats && (
                    <div className="mb-8 bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20">
                        <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
                            <BarChart3 className="w-6 h-6" />
                            Training Performance Overview
                        </h2>

                        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
                            <div className="text-center">
                                <div className="text-3xl font-bold text-green-400 mb-2">
                                    {(stats.finalAccuracy * 100).toFixed(1)}%
                                </div>
                                <div className="text-sm text-gray-300">Final Test Accuracy</div>
                                <div className="text-xs text-gray-400">
                                    (+{(stats.improvement * 100).toFixed(1)}% improvement)
                                </div>
                            </div>

                            <div className="text-center">
                                <div className="text-3xl font-bold text-purple-400 mb-2">
                                    {(stats.bestAccuracy * 100).toFixed(1)}%
                                </div>
                                <div className="text-sm text-gray-300">Best Accuracy</div>
                                <div className="text-xs text-gray-400">
                                    (Epoch {stats.bestEpoch})
                                </div>
                            </div>

                            <div className="text-center">
                                <div className="text-3xl font-bold text-blue-400 mb-2">
                                    {stats.convergence}
                                </div>
                                <div className="text-sm text-gray-300">Convergence</div>
                                <div className="text-xs text-gray-400">
                                    Training Quality
                                </div>
                            </div>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            {/* Training History Chart */}
                            <div className="bg-white/5 rounded-lg p-4">
                                <h3 className="font-semibold mb-3">Accuracy Over Time</h3>
                                <div className="h-32 flex items-end justify-between gap-1">
                                    {trainingHistory.slice(-20).map((point, i) => (
                                        <div
                                            key={i}
                                            className="bg-gradient-to-t from-blue-500 to-purple-500 rounded-t flex-1 min-w-0"
                                            style={{ height: `${point.globalAccuracy * 100}%` }}
                                            title={`Epoch ${point.epoch}: ${(point.globalAccuracy * 100).toFixed(1)}%`}
                                        />
                                    ))}
                                </div>
                            </div>

                            {/* Peer Contributions */}
                            <div className="bg-white/5 rounded-lg p-4">
                                <h3 className="font-semibold mb-3">Peer Contributions</h3>
                                <div className="space-y-2">
                                    {peers.map(peer => (
                                        <div key={peer.id} className="flex items-center justify-between">
                                            <span className="text-sm text-gray-300">{peer.name}</span>
                                            <div className="flex items-center gap-2">
                                                <div className="w-16 bg-gray-700 rounded-full h-2">
                                                    <div
                                                        className="bg-gradient-to-r from-green-500 to-blue-500 h-2 rounded-full"
                                                        style={{ width: `${peer.contribution * 100}%` }}
                                                    />
                                                </div>
                                                <span className="text-xs text-gray-400 w-12">
                                                    {peer.totalUpdates} updates
                                                </span>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                {/* Main Stats */}
                <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
                    <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20">
                        <div className="flex items-center gap-3 mb-2">
                            <Brain className="w-8 h-8 text-purple-400" />
                            <h3 className="text-lg font-semibold">Global Model</h3>
                        </div>
                        <div className="text-3xl font-bold text-purple-400">
                            {(globalAccuracy * 100).toFixed(1)}%
                        </div>
                        <p className="text-sm text-gray-300">Test Accuracy</p>
                    </div>

                    <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20">
                        <div className="flex items-center gap-3 mb-2">
                            <TrendingUp className="w-8 h-8 text-green-400" />
                            <h3 className="text-lg font-semibold">Epoch</h3>
                        </div>
                        <div className="text-3xl font-bold text-green-400">
                            {currentEpoch}
                        </div>
                        <p className="text-sm text-gray-300">
                            {trainingComplete ? 'Completed' : `of ${maxEpochs}`}
                        </p>
                    </div>

                    <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20">
                        <div className="flex items-center gap-3 mb-2">
                            <Users className="w-8 h-8 text-blue-400" />
                            <h3 className="text-lg font-semibold">Active Peers</h3>
                        </div>
                        <div className="text-3xl font-bold text-blue-400">
                            {peers.filter(p => p.status === 'active').length}
                        </div>
                        <p className="text-sm text-gray-300">Connected</p>
                    </div>

                    <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20">
                        <div className="flex items-center gap-3 mb-2">
                            <Clock className="w-8 h-8 text-cyan-400" />
                            <h3 className="text-lg font-semibold">Training Time</h3>
                        </div>
                        <div className="text-3xl font-bold text-cyan-400">
                            {startTime ? formatDuration((endTime || Date.now()) - startTime) : '0s'}
                        </div>
                        <p className="text-sm text-gray-300">Duration</p>
                    </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* Peer Network */}
                    <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20">
                        <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                            <Network className="w-6 h-6" />
                            Peer Network
                        </h3>
                        <div className="space-y-3">
                            {peers.map(peer => (
                                <div key={peer.id} className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
                                    <div className="flex items-center gap-3">
                                        <div className={`${getStatusColor(peer.status)}`}>
                                            {getStatusIcon(peer.status)}
                                        </div>
                                        <div>
                                            <div className="font-semibold">{peer.name}</div>
                                            <div className="text-sm text-gray-400">
                                                {peer.data.length} samples | {(peer.accuracy * 100).toFixed(1)}% local acc
                                            </div>
                                        </div>
                                    </div>
                                    <div className="text-right">
                                        <div className="text-sm text-gray-300">
                                            ↑{peer.uploadSpeed.toFixed(0)} ↓{peer.downloadSpeed.toFixed(0)} KB/s
                                        </div>
                                        <div className="text-xs text-gray-400">
                                            Trust: {(peer.contribution * 100).toFixed(0)}%
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Network Activity */}
                    <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20">
                        <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                            <Zap className="w-6 h-6" />
                            Network Activity
                        </h3>
                        <div className="space-y-2 max-h-64 overflow-y-auto">
                            {networkActivity.slice(-10).map(activity => (
                                <div key={activity.id} className="flex items-center justify-between p-2 bg-white/5 rounded text-sm">
                                    <div className="flex items-center gap-2">
                                        {activity.type === 'model' ?
                                            <Brain className="w-4 h-4 text-purple-400" /> :
                                            <TrendingUp className="w-4 h-4 text-green-400" />
                                        }
                                        <span>
                                            Peer {activity.from} → Peer {activity.to}
                                        </span>
                                    </div>
                                    <div className="flex items-center gap-2 text-gray-300">
                                        <span>{activity.size}KB</span>
                                        {activity.type === 'model' ?
                                            <Upload className="w-3 h-3" /> :
                                            <Download className="w-3 h-3" />
                                        }
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Additional Stats */}
                <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20">
                        <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                            <Database className="w-5 h-5" />
                            Data Statistics
                        </h3>
                        <div className="space-y-2 text-sm">
                            <div className="flex justify-between">
                                <span className="text-gray-300">Training Samples:</span>
                                <span className="font-semibold">{trainingData.length}</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-gray-300">Test Samples:</span>
                                <span className="font-semibold">{testData.length}</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-gray-300">Data Shared:</span>
                                <span className="font-semibold">{(totalDataShared / 1000).toFixed(1)}K</span>
                            </div>
                        </div>
                    </div>

                    <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20">
                        <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                            <Target className="w-5 h-5" />
                            Model Performance
                        </h3>
                        <div className="space-y-2 text-sm">
                            <div className="flex justify-between">
                                <span className="text-gray-300">Best Accuracy:</span>
                                <span className="font-semibold text-green-400">
                                    {stats ? (stats.bestAccuracy * 100).toFixed(1) + '%' : 'N/A'}
                                </span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-gray-300">Improvement:</span>
                                <span className="font-semibold text-blue-400">
                                    {stats ? '+' + (stats.improvement * 100).toFixed(1) + '%' : 'N/A'}
                                </span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-gray-300">Convergence:</span>
                                <span className="font-semibold text-purple-400">
                                    {stats ? stats.convergence : 'N/A'}
                                </span>
                            </div>
                        </div>
                    </div>

                    <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20">
                        <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                            <Network className="w-5 h-5" />
                            Network Stats
                        </h3>
                        <div className="space-y-2 text-sm">
                            <div className="flex justify-between">
                                <span className="text-gray-300">Total Peers:</span>
                                <span className="font-semibold">{peers.length}</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-gray-300">Model Fragments:</span>
                                <span className="font-semibold">{modelFragments}</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-gray-300">Network Updates:</span>
                                <span className="font-semibold">{networkActivity.length}</span>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Training Details */}
                <div className="mt-6 bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20">
                    <h3 className="text-lg font-semibold mb-3">Training Configuration</h3>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                        <div>
                            <span className="text-gray-300">Dataset:</span> {trainingData.length} samples (2D classification)
                        </div>
                        <div>
                            <span className="text-gray-300">Algorithm:</span> Federated Averaging with Gossip Protocol
                        </div>
                        <div>
                            <span className="text-gray-300">Model:</span> Linear Classifier with Sigmoid Activation
                        </div>
                    </div>
                    <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                        <div>
                            <span className="text-gray-300">Max Epochs:</span> {maxEpochs}
                        </div>
                        <div>
                            <span className="text-gray-300">Learning Rate:</span> 0.01
                        </div>
                        <div>
                            <span className="text-gray-300">Blend Factor:</span> 0.3
                        </div>
                    </div>
                </div>

                {/* Footer Info */}
                <div className="mt-8 text-center">
                    <div className="flex items-center justify-center gap-6 text-sm text-gray-400">
                        <span className="flex items-center gap-1">
                            <Shield className="w-4 h-4" />
                            Federated Learning
                        </span>
                        <span className="flex items-center gap-1">
                            <Network className="w-4 h-4" />
                            P2P Protocol
                        </span>
                        <span className="flex items-center gap-1">
                            <Database className="w-4 h-4" />
                            {modelFragments} Model Fragments
                        </span>
                        {trainingComplete && (
                            <span className="flex items-center gap-1 text-green-400">
                                <Award className="w-4 h-4" />
                                Training Complete
                            </span>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default MLTorrent;